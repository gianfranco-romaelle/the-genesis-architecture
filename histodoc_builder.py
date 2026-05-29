#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
histodoc_builder.py — extract three-layer HistoDoc JSON from PDF sources.

Three layers per region (see schema/HistoDoc.ts):
  diplomatic   raw extracted text — original hyphenation, spelling, spacing
  normalized   cleaned text — hyphenation fixed, NFC, whitespace collapsed
  translation  English rendering; for English sources == normalized
               (non-English translation deferred to future pipeline pass)

Output: histodoc/<safe_stem>.histodoc.json
State:  histodoc_state.json  (keyed by source_path; atomic writes)

Usage
  python histodoc_builder.py                   # all pending pipeline-indexed files
  python histodoc_builder.py path/to/file.pdf  # single file (explicit)
  python histodoc_builder.py --limit 20        # first N pending
  python histodoc_builder.py --reset           # re-process all
  python histodoc_builder.py --source-lang lat # filter source language
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

try:
    import fitz  # PyMuPDF
    FITZ_OK = True
except ImportError:
    FITZ_OK = False
    print("WARNING: PyMuPDF (fitz) not installed — cannot process PDFs", flush=True)

_HERE            = Path(__file__).parent
_STATE_PATH      = _HERE / "histodoc_state.json"
_OUT_DIR         = _HERE / "histodoc"
_PIPELINE_STATE  = _HERE / "pipeline_state.json"
SCHEMA_VERSION   = 1

# ── state helpers ─────────────────────────────────────────────────────────────

def _load_state() -> dict:
    try:
        return json.loads(_STATE_PATH.read_bytes().decode("utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {"files": {}}


def _save_state(state: dict) -> None:
    tmp = _STATE_PATH.with_suffix(".tmp")
    tmp.write_bytes(
        json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8")
    )
    os.replace(tmp, _STATE_PATH)


def _update_entry(state: dict, source_path: str, entry: dict) -> None:
    state.setdefault("files", {})[source_path] = entry
    _save_state(state)


# ── region classification (heuristic, PyMuPDF blocks) ─────────────────────────

def _classify_block(blk: dict, pw: float, ph: float) -> str:
    """Infer RegionType from PyMuPDF block geometry and font metrics."""
    x0, y0, x1, y1 = blk["bbox"]

    # running title: top or bottom 8 % of page
    if y1 < ph * 0.08 or y0 > ph * 0.92:
        return "running_title"

    spans = [s for ln in blk.get("lines", []) for s in ln.get("spans", [])]
    texts = [s for s in spans if s.get("text", "").strip()]
    if not texts:
        return "paragraph"

    sizes = sorted(s["size"] for s in texts)
    median_size = sizes[len(sizes) // 2]
    max_size    = sizes[-1]

    # footnote: small text near lower third of page
    if median_size <= 8.5 and y0 > ph * 0.70:
        return "footnote"
    if median_size <= 7.5:
        return "footnote"

    # heading: large font
    if max_size >= 14.0:
        return "heading"

    # subheading: medium bold text
    if max_size >= 11.5 and any(s.get("flags", 0) & 16 for s in texts):
        return "subheading"

    # margin note: very narrow block at horizontal edges
    block_w = x1 - x0
    if block_w < pw * 0.22 and (x0 < pw * 0.07 or x1 > pw * 0.93):
        return "margin_note"

    # caption: small text directly following image block or matching "Fig." / "Table"
    raw = " ".join(s.get("text", "") for s in texts)
    if median_size <= 9.5 and re.match(r"(Fig\.|Table|Plate|Tableau)\s", raw, re.I):
        return "caption"

    return "paragraph"


# ── text extraction ───────────────────────────────────────────────────────────

def _diplomatic_text(blk: dict) -> str:
    """Raw block text preserving original hyphenation and line structure."""
    lines = []
    for ln in blk.get("lines", []):
        line = "".join(s.get("text", "") for s in ln.get("spans", []))
        if line.strip():
            lines.append(line)
    return "\n".join(lines)


_HYPHEN_RE  = re.compile(r"(\w)-\n(\w)")   # end-of-line hyphenation
_SOFTWR_RE  = re.compile(r"(?<!\n)\n(?!\n)")  # single newline (soft wrap)
_SPACES_RE  = re.compile(r"[ \t]{2,}")


def _normalize(text: str) -> str:
    """Normalized layer: NFC, hyphenation fix, whitespace collapse."""
    text = unicodedata.normalize("NFC", text)
    # fix hyphenated words broken across lines
    text = _HYPHEN_RE.sub(r"\1\2", text)
    # collapse soft-wrapped lines into single space
    text = _SOFTWR_RE.sub(" ", text)
    # collapse runs of spaces/tabs
    text = _SPACES_RE.sub(" ", text)
    return text.strip()


# ── document processing ───────────────────────────────────────────────────────

def _safe_stem(path: Path) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", path.stem)[:120]


def process_file(
    src_path: Path,
    source_lang: str = "eng",
    source_file_id: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """
    Build a HistoDocDocument dict from a PDF.
    Returns the document dict on success, raises on failure.
    """
    if not FITZ_OK:
        raise RuntimeError("PyMuPDF not available")

    doc = fitz.open(str(src_path))
    regions: list[dict] = []
    layers:  list[dict] = []
    seq = 0
    doc_id = str(uuid.uuid4())

    for page_num, page in enumerate(doc, start=1):
        pw, ph = page.rect.width, page.rect.height
        blocks = page.get_text(
            "dict", flags=fitz.TEXT_PRESERVE_WHITESPACE
        ).get("blocks", [])

        for blk in blocks:
            if blk.get("type", 0) != 0:  # skip image blocks
                continue
            dipl = _diplomatic_text(blk)
            if not dipl.strip():
                continue

            region_type = _classify_block(blk, pw, ph)
            rid = str(uuid.uuid4())

            regions.append({
                "canonical_region_id": rid,
                "page_number":         page_num,
                "region_type":         region_type,
                "sequence_index":      seq,
                "bbox":                list(blk["bbox"]),
                "block_id":            str(blk.get("number", seq)),
            })

            norm = _normalize(dipl)

            # diplomatic layer
            layers.append({
                "canonical_region_id": rid,
                "layer_type":          "diplomatic",
                "text":                dipl,
                "language":            source_lang,
            })

            # normalized layer
            layers.append({
                "canonical_region_id": rid,
                "layer_type":          "normalized",
                "text":                norm,
                "language":            source_lang,
            })

            # translation layer
            # for English sources, translation == normalized (no API needed)
            # for non-English sources, mark as pending with empty text
            if source_lang == "eng":
                layers.append({
                    "canonical_region_id": rid,
                    "layer_type":          "translation",
                    "text":                norm,
                    "language":            "eng",
                    "translation_note":    "source language is English; translation == normalized",
                })
            else:
                layers.append({
                    "canonical_region_id": rid,
                    "layer_type":          "translation",
                    "text":                "",
                    "language":            "eng",
                    "translation_note":    "pending — non-English translation deferred",
                })

            seq += 1

    page_count = doc.page_count
    doc.close()

    # infer LaTeX template
    if len(regions) > 400:
        tmpl = "memoir"
    elif any(r["region_type"] in ("footnote", "margin_note") for r in regions):
        tmpl = "reledpar"
    else:
        tmpl = "article"

    now = datetime.now(timezone.utc).isoformat()
    md  = metadata or {}

    return {
        "document_id":    doc_id,
        "schema_version": SCHEMA_VERSION,
        "source_path":    str(src_path),
        "source_file_id": source_file_id,
        "source_language": source_lang,
        "title":          md.get("title"),
        "author":         md.get("author"),
        "year":           md.get("year"),
        "page_count":     page_count,
        "regions":        regions,
        "layers":         layers,
        "latex_template": tmpl,
        "created_at":     now,
        "updated_at":     now,
    }


def save_histodoc(doc: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem    = _safe_stem(Path(doc["source_path"]))
    out     = out_dir / f"{stem}.histodoc.json"
    tmp     = out.with_suffix(".tmp")
    tmp.write_bytes(json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8"))
    os.replace(tmp, out)
    return out


# ── pipeline-state integration ────────────────────────────────────────────────

def _indexed_files() -> list[str]:
    """Return source paths from pipeline_state.json with status=='indexed'."""
    try:
        data = json.loads(
            _PIPELINE_STATE.read_bytes().decode("utf-8", errors="replace")
        )
        return [
            p for p, v in data.get("files", {}).items()
            if v.get("status") == "indexed"
        ]
    except Exception:
        return []


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    ap = argparse.ArgumentParser(description="HistoDoc three-layer extractor")
    ap.add_argument(
        "files", nargs="*",
        help="Explicit PDF paths. Omit to use indexed files from pipeline_state.json",
    )
    ap.add_argument("--limit", type=int, default=None, help="Process at most N files")
    ap.add_argument("--reset", action="store_true", help="Re-process already-done files")
    ap.add_argument(
        "--source-lang", default="eng",
        help="ISO 639-3 source language (default: eng). "
             "Non-English translation is deferred.",
    )
    ap.add_argument(
        "--out-dir", default=str(_OUT_DIR),
        help=f"Output directory for .histodoc.json files (default: {_OUT_DIR})",
    )
    return ap.parse_args()


def main():
    args     = _parse_args()
    state    = _load_state()
    out_dir  = Path(args.out_dir)
    done_set = set(state.get("files", {}).keys())

    # collect candidates
    if args.files:
        candidates = [Path(f) for f in args.files]
    else:
        candidates = [Path(p) for p in _indexed_files()]

    if not args.reset:
        candidates = [p for p in candidates if str(p) not in done_set]

    if args.limit:
        candidates = candidates[: args.limit]

    print(f"HistoDoc builder: {len(candidates)} files to process", flush=True)
    if not candidates:
        print("Nothing to do.", flush=True)
        return

    ok = err = skip = 0
    for i, src in enumerate(candidates, 1):
        src_str = str(src)
        if not src.exists():
            print(f"  [{i}/{len(candidates)}] SKIP (not found): {src.name}", flush=True)
            skip += 1
            continue
        if src.suffix.lower() not in (".pdf",):
            print(f"  [{i}/{len(candidates)}] SKIP (unsupported ext): {src.name}", flush=True)
            skip += 1
            continue

        print(f"  [{i}/{len(candidates)}] {src.name} ... ", end="", flush=True)
        try:
            doc      = process_file(src, source_lang=args.source_lang)
            out_path = save_histodoc(doc, out_dir)
            n_reg    = len(doc["regions"])
            n_lay    = len(doc["layers"])
            layer_counts = {
                lt: sum(1 for l in doc["layers"] if l["layer_type"] == lt)
                for lt in ("diplomatic", "normalized", "translation")
            }
            _update_entry(state, src_str, {
                "document_id":   doc["document_id"],
                "source_path":   src_str,
                "status":        "parsed" if args.source_lang == "eng" else "parsed",
                "region_count":  n_reg,
                "layer_counts":  layer_counts,
                "latex_path":    None,
                "error":         None,
                "updated_at":    doc["updated_at"],
            })
            print(f"{n_reg} regions, {n_lay} layers → {out_path.name}", flush=True)
            ok += 1
        except Exception as exc:
            print(f"ERROR: {exc}", flush=True)
            _update_entry(state, src_str, {
                "document_id":  "",
                "source_path":  src_str,
                "status":       "failed",
                "region_count": 0,
                "layer_counts": {"diplomatic": 0, "normalized": 0, "translation": 0},
                "latex_path":   None,
                "error":        str(exc),
                "updated_at":   datetime.now(timezone.utc).isoformat(),
            })
            err += 1

    print(f"\nDone: {ok} parsed, {err} failed, {skip} skipped", flush=True)


if __name__ == "__main__":
    main()
