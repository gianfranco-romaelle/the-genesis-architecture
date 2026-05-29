#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
graph_builder.py — HomΩ first-pass ontology extractor

Reads pipeline_state.json to find indexed source files, parses their
text with PyMuPDF, and sends a structured extraction prompt to an
OpenRouter free model.  Results are written to graph_state.json.

Usage
  python graph_builder.py [options]

Options
  --limit N            process at most N files per batch (0 = all)
  --reset              clear graph_state.json and reprocess all files
  --file PATH          process a single specific file (relative to library root)
  --queue              consume pipeline_queue.json graph_queue
  --watch              daemon mode: stay running and pick up newly indexed files
  --watch-interval N   seconds between pipeline_state polls (default: 30)
  --model MODEL        OpenRouter model slug  (default: meta-llama/llama-3.3-70b-instruct:free)
  --dry-run            parse and print extraction prompt only, no API calls
  --delay N            seconds between API calls to respect rate limits (default: 3)
"""
import sys, os, json, re, time, argparse, traceback
from pathlib import Path

# ensure UTF-8 output even when stdout is not a TTY (e.g. Start-Process redirect)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass  # Python < 3.7 fallback
from datetime import datetime, timezone
from typing import Optional

# ── paths ─────────────────────────────────────────────────────────────────────

_HERE           = Path(__file__).parent
_PIPELINE_STATE = _HERE / "pipeline_state.json"
_GRAPH_STATE    = _HERE / "graph_state.json"
_QUEUE_PATH     = _HERE / "pipeline_queue.json"
_ENV_PATH       = _HERE / ".env"
_CONFIG_PATH    = _HERE / "throttle_config.json"

# ── schema constants ──────────────────────────────────────────────────────────

EVIDENCE_KINDS = (
    "in_source_text_direct",
    "secondary_source_inferred",
    "llm_training_knowledge",
    "llm_prior_pass_recalled",
    "research_task_queued",
)

ERA_MAP = {
    0: ("Calc0", "pre-1600"),
    1: ("Calc1", "1600-1680"),
    2: ("Calc2", "1680-1730"),
    3: ("Calc3", "1730-1800"),
    4: ("Calc4", "1800-1870"),
    5: ("Calc5", "1870-1950"),
    6: ("Calc6", "1950-present"),
}

DEFAULT_MODEL   = "meta-llama/llama-3.3-70b-instruct:free"
TEXT_CHAR_LIMIT = 6000   # ~1500 tokens; keep prompts fast for free-tier models
MAX_RETRIES       = 4
RETRY_BASE_SECS   = 15     # backoff: 15 s, 30 s, 60 s, 120 s  (free-tier rate limits need ~60-120 s)
MAX_GRAPH_RETRIES = 5      # give up on a file after this many consecutive API failures

# Fallback chain: tried in order when the primary model exhausts all retries with 429
FALLBACK_MODELS = [
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "openai/gpt-oss-20b:free",
    "deepseek/deepseek-v4-flash:free",
]

# ── env loading ───────────────────────────────────────────────────────────────

def _load_env():
    if _ENV_PATH.exists():
        for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

_load_env()

# ── text extraction ───────────────────────────────────────────────────────────

def extract_text(filepath: Path, char_limit: int = TEXT_CHAR_LIMIT) -> Optional[str]:
    try:
        import fitz  # PyMuPDF
        doc  = fitz.open(str(filepath))
        text = ""
        for page in doc:
            text += page.get_text()
            if len(text) >= char_limit:
                break
        doc.close()
        return text[:char_limit].strip() or None
    except Exception as exc:
        print(f"  [text] PyMuPDF failed for {filepath.name}: {exc}", flush=True)
        return None


def extract_text_fallback(filepath: Path, char_limit: int = TEXT_CHAR_LIMIT) -> Optional[str]:
    try:
        return filepath.read_text(encoding="utf-8", errors="replace")[:char_limit].strip() or None
    except Exception:
        return None


def get_text(filepath: Path) -> Optional[str]:
    suffix = filepath.suffix.lower()
    if suffix == ".pdf":
        return extract_text(filepath) or extract_text_fallback(filepath)
    return extract_text_fallback(filepath)


# ── extraction prompt ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert historian of mathematics and science specialising in the
Auguste Laurent Society (ALS) research framework.  Your task is to extract
structured entities from archival text according to the HomΩ ontology.

HomΩ era assignments (calculus_number):
  0 = Calc0  pre-1600
  1 = Calc1  1600–1680   (Kepler, Galileo, Descartes, Fermat)
  2 = Calc2  1680–1730   (Newton, Leibniz, Bernoullis)
  3 = Calc3  1730–1800   (Euler, Lagrange, d'Alembert)
  4 = Calc4  1800–1870   (Cauchy, Gauss, Riemann, Fourier, Laurent)
  5 = Calc5  1870–1950   (Poincaré, Hilbert, Cantor, Weierstrass)
  6 = Calc6  1950–present

Evidence kinds:
  in_source_text_direct       — person or relationship stated in this text
  secondary_source_inferred   — inferable from context but not stated
  llm_training_knowledge      — you know this from training, not from this text

Respond ONLY with a JSON object.  No explanation, no markdown fences."""

_USER_TEMPLATE = """\
Extract all persons and their relationships from the following text.

Return a JSON object with exactly this structure:
{{
  "persons": [
    {{
      "name": "Full Name",
      "birth_year": 1789,
      "death_year": 1857,
      "calculus_number": 4,
      "nationality": "French",
      "occupation": "mathematician",
      "evidence": "in_source_text_direct"
    }}
  ],
  "relationships": [
    {{
      "subject": "Person A name",
      "predicate": "short verb phrase",
      "object": "Person B name",
      "evidence": "in_source_text_direct"
    }}
  ]
}}

Rules:
- Only include persons actually present in or directly inferable from the text.
- calculus_number is the HomΩ era (0–6) based on their active period.
- birth_year / death_year may be null if unknown.
- RELATIONSHIPS MUST BE PERSON-TO-PERSON: both subject and object must be person names.
  Good: "studied under", "collaborated with", "corresponded with", "was student of",
        "influenced", "co-authored with", "mentored", "succeeded", "refuted work of".
  Skip: person→concept edges ("developed X", "introduced Y") — those are not useful here.
- Limit to the 15 most significant persons and 20 most significant person-to-person relationships.

TEXT:
---
{text}
---"""


def build_prompt(text: str) -> str:
    return _USER_TEMPLATE.format(text=text[:TEXT_CHAR_LIMIT])


# ── OpenRouter API with retry / backoff ───────────────────────────────────────

def _call_one_model(prompt: str, model: str, api_key: str) -> Optional[dict]:
    """Attempt a single model with exponential backoff. Returns dict or None."""
    import openai
    for attempt in range(MAX_RETRIES):
        try:
            client = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2500,
                timeout=90,
            )
            choices = resp.choices if resp and resp.choices else []
            if not choices:
                print(f"  [api] empty choices in response", flush=True)
                return None
            raw = choices[0].message.content or ""
            return _parse_response(raw)

        except openai.RateLimitError:
            wait = RETRY_BASE_SECS * (2 ** attempt)
            print(f"  [api] 429 rate limit — retry {attempt+1}/{MAX_RETRIES} in {wait}s",
                  flush=True)
            time.sleep(wait)

        except openai.APIStatusError as exc:
            code = getattr(exc, "status_code", None)
            if code and code >= 500:
                wait = RETRY_BASE_SECS * (2 ** attempt)
                print(f"  [api] {code} server error — retry {attempt+1}/{MAX_RETRIES} in {wait}s",
                      flush=True)
                time.sleep(wait)
            else:
                print(f"  [api] error {code}: {exc}", flush=True)
                return None

        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BASE_SECS * (2 ** attempt)
                print(f"  [api] error: {exc} — retry {attempt+1}/{MAX_RETRIES} in {wait}s",
                      flush=True)
                time.sleep(wait)
            else:
                print(f"  [api] fatal: {exc}", flush=True)
                return None

    return None  # all retries exhausted for this model


def call_openrouter(text: str, model: str, dry_run: bool = False) -> Optional[dict]:
    """Call OpenRouter with retry/backoff; auto-falls back through FALLBACK_MODELS on 429 exhaustion."""
    prompt = build_prompt(text)

    if dry_run:
        print("\n-- EXTRACTION PROMPT ------------------------------------------")
        print(f"System: {_SYSTEM_PROMPT[:200]}")
        print(f"User:   {prompt[:400]}")
        print("-- END PROMPT -------------------------------------------------\n")
        return {"persons": [], "relationships": [], "_dry_run": True}

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("  [api] OPENROUTER_API_KEY not set — skipping", flush=True)
        return None

    # Build model chain: primary first, then fallbacks (deduped, preserving order)
    chain = [model] + [m for m in FALLBACK_MODELS if m != model]

    for m in chain:
        print(f"  [api] trying {m}", flush=True)
        result = _call_one_model(prompt, m, api_key)
        if result is not None:
            return result
        print(f"  [api] {m} exhausted — trying next fallback", flush=True)

    print("  [api] all models exhausted", flush=True)
    return None


def _parse_response(raw: str) -> Optional[dict]:
    """Extract JSON from model response, tolerating markdown fences."""
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if not m:
        print(f"  [parse] No JSON found in response: {raw[:200]}", flush=True)
        return None
    try:
        data = json.loads(m.group())
        data.setdefault("persons", [])
        data.setdefault("relationships", [])
        for obj in data["persons"] + data["relationships"]:
            if obj.get("evidence") not in EVIDENCE_KINDS:
                obj["evidence"] = "llm_training_knowledge"
        data["persons"] = _dedup_persons(data["persons"])
        _fix_calculus_numbers(data["persons"])
        _filter_person_person_rels(data)
        return data
    except json.JSONDecodeError as exc:
        print(f"  [parse] JSON decode error: {exc} — raw: {raw[:200]}", flush=True)
        return None


def _filter_person_person_rels(data: dict) -> None:
    """Drop relationships whose object is not a name in the extracted persons list."""
    person_names = {p.get("name", "").strip().lower() for p in data.get("persons", []) if p.get("name")}
    before = len(data.get("relationships", []))
    data["relationships"] = [
        r for r in data.get("relationships", [])
        if r.get("object", "").strip().lower() in person_names
    ]
    dropped = before - len(data["relationships"])
    if dropped:
        print(f"  [filter] dropped {dropped} person→concept relationship(s)", flush=True)


def _dedup_persons(persons: list) -> list:
    """Remove duplicate persons by case-folded name, keeping first occurrence."""
    seen: set = set()
    result = []
    for p in persons:
        key = p.get("name", "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(p)
    return result


def _fix_calculus_numbers(persons: list):
    """Correct obvious era misassignments based on birth year.

    Active period ≈ birth_year + 20, so thresholds are shifted back 20 years
    from the era boundaries to reflect when a person would have been working.
      born >= 1930 → active from ~1950 → Calc6 minimum
      born >= 1850 → active from ~1870 → Calc5 minimum
      born >= 1780 → active from ~1800 → Calc4 minimum
    """
    for p in persons:
        by = p.get("birth_year")
        cn = p.get("calculus_number")
        if by is None or cn is None:
            continue
        try:
            by, cn = int(by), int(cn)
        except (TypeError, ValueError):
            continue
        if by >= 1930 and cn < 6:
            p["calculus_number"] = 6
        elif by >= 1850 and cn < 5:
            p["calculus_number"] = 5
        elif by >= 1780 and cn < 4:
            p["calculus_number"] = 4


# ── state management ──────────────────────────────────────────────────────────

def load_graph_state() -> dict:
    try:
        return json.loads(_GRAPH_STATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": "1.0", "files": {}}


def save_graph_state(state: dict):
    """Atomic write via .tmp → os.replace to survive kill mid-write."""
    tmp = _GRAPH_STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, _GRAPH_STATE)


def load_pipeline_state() -> dict:
    try:
        return json.loads(_PIPELINE_STATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"files": {}}


def load_queue() -> list:
    try:
        q = json.loads(_QUEUE_PATH.read_text(encoding="utf-8"))
        return q.get("graph_queue", [])
    except (OSError, json.JSONDecodeError):
        return []


def dequeue(path: str):
    try:
        q = json.loads(_QUEUE_PATH.read_text(encoding="utf-8"))
        q["graph_queue"] = [e for e in q.get("graph_queue", []) if e.get("path") != path]
        _QUEUE_PATH.write_text(json.dumps(q, indent=2), encoding="utf-8")
    except (OSError, json.JSONDecodeError):
        pass


def era_distribution(persons: list) -> dict:
    dist: dict = {}
    for p in persons:
        cn = p.get("calculus_number")
        if cn is not None:
            key = ERA_MAP.get(int(cn), (f"Calc{cn}", ""))[0]
            dist[key] = dist.get(key, 0) + 1
    return dist


# ── config / library root ─────────────────────────────────────────────────────

def get_library_root() -> Optional[Path]:
    # 1. throttle_config.json (canonical source)
    try:
        cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        root = cfg.get("library_root")
        if root:
            p = Path(root)
            if p.exists():
                return p
    except Exception:
        pass
    # 2. pipeline_state.json top-level key (for future use)
    try:
        ps = load_pipeline_state()
        root = ps.get("library_root")
        if root:
            p = Path(root)
            if p.exists():
                return p
    except Exception:
        pass
    # 3. hard-coded fallbacks
    for candidate in [
        Path(r"G:\Other computers\My Laptop\THE AUGUSTE LAURENT SOCIETY"),
        _HERE / "library",
    ]:
        if candidate.exists():
            return candidate
    return None


# ── file processing ───────────────────────────────────────────────────────────

def process_file(rel_path: str, library_root: Path, model: str,
                 dry_run: bool, delay: float) -> dict:
    abs_path = library_root / rel_path
    entry: dict = {
        "status":                  "error",
        "objects_extracted":       0,
        "relationships_extracted": 0,
        "era_distribution":        {},
        "persons":                 [],
        "relationships":           [],
        "updated_at":              datetime.now(timezone.utc).isoformat(),
        "error":                   None,
    }

    if not abs_path.exists():
        entry["error"] = f"file not found: {abs_path}"
        print(f"  [skip] {abs_path.name}: not found", flush=True)
        return entry

    print(f"  [read] {abs_path.name}", flush=True)
    text = get_text(abs_path)
    if not text:
        entry["error"] = "no text extracted"
        return entry

    print(f"  [chars] {len(text):,}", flush=True)
    result = call_openrouter(text, model=model, dry_run=dry_run)

    if result is None:
        entry["error"] = "api call failed"
        return entry

    persons       = result.get("persons", [])
    relationships = result.get("relationships", [])

    entry.update({
        "status":                  "complete",
        "objects_extracted":       len(persons),
        "relationships_extracted": len(relationships),
        "era_distribution":        era_distribution(persons),
        "persons":                 persons,
        "relationships":           relationships,
        "error":                   None,
    })

    print(f"  [ok]   {len(persons)} persons  {len(relationships)} relationships",
          flush=True)

    if not dry_run:
        time.sleep(delay)

    return entry


def _get_pending(graph_state: dict, args) -> list:
    if args.file:
        return [args.file]
    if args.queue:
        return [e["path"] for e in
                sorted(load_queue(), key=lambda e: -e.get("priority", 0))]
    pipeline_state = load_pipeline_state()
    _PROCESSABLE = {"indexed", "embedded_no_qdrant"}
    indexed = {p for p, info in pipeline_state.get("files", {}).items()
               if info.get("status") in _PROCESSABLE}
    done = {p for p, info in graph_state["files"].items()
            if info.get("status") == "complete"
            or info.get("retry_count", 0) >= MAX_GRAPH_RETRIES}
    return sorted(indexed - done)


def run_batch(args, library_root: Path, graph_state: dict) -> int:
    """Process one batch of pending files.  Returns count processed."""
    to_process = _get_pending(graph_state, args)

    if not to_process:
        return 0

    limit = args.limit if args.limit else len(to_process)
    batch = to_process[:limit]

    print(f"Files to extract: {len(batch)} / {len(to_process)} pending", flush=True)
    print(flush=True)

    processed = 0
    for rel_path in batch:
        processed += 1
        print(f"[{processed}/{len(batch)}] {rel_path}", flush=True)
        try:
            entry = process_file(rel_path, library_root, args.model,
                                 args.dry_run, args.delay)
            if entry.get("status") == "error":
                prev_retries = graph_state["files"].get(rel_path, {}).get("retry_count", 0)
                entry["retry_count"] = prev_retries + 1
                if entry["retry_count"] >= MAX_GRAPH_RETRIES:
                    print(f"  [abandon] {rel_path}: {MAX_GRAPH_RETRIES} failures — skipping permanently",
                          flush=True)
            graph_state["files"][rel_path] = entry
        except Exception:
            traceback.print_exc()
            prev_retries = graph_state["files"].get(rel_path, {}).get("retry_count", 0)
            graph_state["files"][rel_path] = {
                "status":      "error",
                "error":       traceback.format_exc(limit=3),
                "retry_count": prev_retries + 1,
                "updated_at":  datetime.now(timezone.utc).isoformat(),
            }

        if not args.dry_run:
            save_graph_state(graph_state)

        if args.queue:
            dequeue(rel_path)

    return processed


def _run_enrich():
    """Run enrich.py as a subprocess to regenerate sacred_timeline_enriched.json."""
    import subprocess
    enrich_script = _HERE / "enrich.py"
    if not enrich_script.exists():
        print("[enrich] enrich.py not found — skipping", flush=True)
        return
    print("[enrich] regenerating sacred_timeline_enriched.json ...", flush=True)
    try:
        r = subprocess.run(
            [sys.executable, "-u", str(enrich_script)],
            capture_output=True, text=True, timeout=120,
        )
        for line in r.stdout.strip().splitlines():
            print(f"[enrich] {line}", flush=True)
        if r.returncode != 0:
            print(f"[enrich] exit {r.returncode}: {r.stderr[:200]}", flush=True)
    except Exception as exc:
        print(f"[enrich] failed: {exc}", flush=True)


def _watch_loop(args, library_root: Path, graph_state: dict):
    interval = args.watch_interval
    enrich = getattr(args, "enrich", False)
    print(f"Watch mode: polling every {interval}s.  Ctrl-C to stop.", flush=True)
    if enrich:
        print("Auto-enrich: will regenerate sacred_timeline_enriched.json after each batch.",
              flush=True)
    total = 0
    while True:
        n = run_batch(args, library_root, graph_state)
        total += n
        if n:
            print(f"[watch] batch done.  Session total: {total}", flush=True)
            if enrich and not args.dry_run:
                _run_enrich()
        else:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[watch] {ts}  nothing new — sleeping {interval}s", flush=True)
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n[watch] stopped.  Session total: {total}", flush=True)
            break


# ── entry point ───────────────────────────────────────────────────────────────

def run(args):
    library_root = get_library_root()
    if library_root is None:
        print("ERROR: cannot locate library root.  Add library_root to throttle_config.json "
              "or ensure G: drive is mounted.", flush=True)
        sys.exit(1)

    print(f"Library root : {library_root}", flush=True)
    print(f"Model        : {args.model}", flush=True)
    if args.dry_run:
        print("Dry-run mode : no API calls will be made", flush=True)

    graph_state = (load_graph_state() if not args.reset
                   else {"schema_version": "1.0", "files": {}})

    if args.watch:
        _watch_loop(args, library_root, graph_state)
    else:
        n = run_batch(args, library_root, graph_state)
        if n == 0 and not args.file and not args.queue:
            print("Nothing to process.  All indexed files already have graph entries.",
                  flush=True)
        print(f"\nDone.  {n} files processed.", flush=True)
        if args.dry_run:
            print("(dry-run — graph_state.json was NOT written)", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit",          type=int,   default=0,
                        help="max files per batch (0 = all)")
    parser.add_argument("--reset",          action="store_true",
                        help="clear graph_state.json before starting")
    parser.add_argument("--file",           type=str,   default="",
                        help="process a single file (relative path from library root)")
    parser.add_argument("--queue",          action="store_true",
                        help="consume pipeline_queue.json graph_queue")
    parser.add_argument("--watch",          action="store_true",
                        help="daemon mode: keep running, pick up newly indexed files")
    parser.add_argument("--enrich",         action="store_true",
                        help="auto-run enrich.py after each successful batch")
    parser.add_argument("--watch-interval", type=int,   default=30,
                        dest="watch_interval",
                        help="seconds between polls in watch mode (default: 30)")
    parser.add_argument("--model",          type=str,   default=DEFAULT_MODEL,
                        help=f"OpenRouter model  (default: {DEFAULT_MODEL})")
    parser.add_argument("--dry-run",        action="store_true",
                        help="print prompts, make no API calls, write no files")
    parser.add_argument("--delay",          type=float, default=12.0,
                        help="seconds between API calls  (default: 12)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
