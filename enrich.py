#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
enrich.py — merge graph_state.json into sacred_timeline for Constellation

Reads sacred_timeline_current.json (flat array of persons) and graph_state.json
(persons + relationships extracted from library documents), and writes
sacred_timeline_enriched.json as { "people": [...] }.

Persons extracted by graph_builder are:
  - Matched by normalized name to existing timeline entries and their
    relationships added as semantic_relationships.
  - If unmatched, appended as new PersonRecord entries with source provenance.

Usage
  python enrich.py [--timeline PATH] [--graph PATH] [--out PATH] [--dry-run]
"""
import sys, os, json, re, unicodedata, argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE          = Path(__file__).parent
_GRAPH_STATE   = _HERE / "graph_state.json"
_ENRICHED_OUT  = _HERE / "sacred_timeline_enriched.json"

_TIMELINE_CANDIDATES = [
    _HERE / "sacred_timeline_current.json",
    _HERE.parent / "sacred-timeline" / "public" / "sacred_timeline_current.json",
]

ERA_MAP = {
    0: "Zeroth", 1: "First",  2: "Second", 3: "Third",
    4: "Fourth", 5: "Fifth",  6: "Sixth",
}


# ── name normalisation ────────────────────────────────────────────────────────

def _norm_name(name: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    nfd = unicodedata.normalize("NFD", name)
    ascii_only = nfd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_only).strip().lower()


def _build_name_index(persons: list) -> dict:
    """Return {normalized_name: person_dict} for fast lookup."""
    index: dict = {}
    for p in persons:
        key = _norm_name(p.get("name", ""))
        if key:
            index[key] = p
    return index


def _find_by_suffix(name: str, index: dict) -> Optional[dict]:
    """
    Fallback: match extracted name as a word-suffix of a timeline name.
    'von Neumann' → matches 'john von neumann' (suffix match, 1 extra word).
    Returns the matched person only if exactly one unambiguous match exists.
    Requires the extracted name to be ≥2 words to avoid single-surname collisions.
    """
    norm = _norm_name(name)
    words = norm.split()
    if len(words) < 2:
        return None  # single-word names are too ambiguous
    candidates = [p for key, p in index.items()
                  if key.endswith(" " + norm) or key.endswith(norm)]
    if len(candidates) == 1:
        return candidates[0]
    return None  # 0 or multiple → no safe match


# ── semantic relationship conversion ─────────────────────────────────────────

def _to_semantic_rel(rel: dict, provenance: str) -> dict:
    """Convert graph_builder relationship dict to SemanticRelationship format."""
    return {
        "source":          rel.get("subject", ""),
        "target":          rel.get("object",  ""),
        "relation_type":   rel.get("predicate", "related to"),
        "weight":          1.0,
        "confidence":      "low",
        "evidence_type":   rel.get("evidence", "llm_training_knowledge"),
        "provenance":      provenance,
        "assertion_layer": "ai_hypothesis",
    }


# ── merge logic ───────────────────────────────────────────────────────────────

def enrich(timeline_persons: list, graph_state: dict, verbose: bool = False) -> dict:
    """
    Merge graph_state into timeline_persons.
    Returns {"people": [...], "suggested_people_registry": [...],
             "ai_relationships": [...], "suggested_tags_registry": []}.
    Persons list is original persons mutated in place + new persons appended.
    """
    name_index = _build_name_index(timeline_persons)
    new_persons: list = []
    suggested_people: list = []
    all_ai_rels: list = []
    stats = {"matched": 0, "new": 0, "rels_added": 0, "files": 0}

    for rel_path, entry in graph_state.get("files", {}).items():
        if entry.get("status") != "complete":
            continue

        stats["files"] += 1
        file_name = Path(rel_path).name
        provenance = f"graph_builder:{file_name}"

        extracted_persons = entry.get("persons", [])
        relationships     = entry.get("relationships", [])

        # Index extracted persons by name for relationship lookup
        extracted_index = _build_name_index(extracted_persons)

        # Upsert each extracted person into the timeline
        for ep in extracted_persons:
            key = _norm_name(ep.get("name", ""))
            if not key:
                continue

            if key in name_index:
                existing = name_index[key]
            else:
                existing = _find_by_suffix(ep.get("name", ""), name_index)
                if existing and verbose:
                    print(f"  [suffix] {ep['name']} -> {existing['name']}", flush=True)

            if existing is not None:
                # Matched (exact or suffix) — annotate with provenance
                if "source_files" not in existing:
                    existing["source_files"] = []
                if file_name not in existing["source_files"]:
                    existing["source_files"].append(file_name)
                # Re-index under extracted name so relationship lookup works
                if key not in name_index:
                    name_index[key] = existing
                stats["matched"] += 1
                if verbose and key in name_index:
                    print(f"  [match] {ep['name']}", flush=True)
            else:
                # New person — create a timeline entry
                new_entry = {
                    "name":            ep.get("name", ""),
                    "birth_year":      ep.get("birth_year"),
                    "death_year":      ep.get("death_year"),
                    "country":         ep.get("nationality"),
                    "field":           ep.get("occupation"),
                    "calculus_number": ep.get("calculus_number"),
                    "calculus_name":   ERA_MAP.get(ep.get("calculus_number", -1), ""),
                    "source_files":    [file_name],
                    "semantic_relationships": [],
                }
                new_persons.append(new_entry)
                name_index[key] = new_entry
                stats["new"] += 1
                # Surface in review queue
                note_parts = []
                if ep.get("birth_year"):
                    note_parts.append(f"b. {ep['birth_year']}")
                if ep.get("occupation"):
                    note_parts.append(ep["occupation"])
                suggested_people.append({
                    "name":         ep.get("name", ""),
                    "source_title": file_name,
                    "note":         ", ".join(note_parts) if note_parts else None,
                })
                if verbose:
                    print(f"  [new]   {ep['name']}", flush=True)

        # Add relationships to the source person's semantic_relationships
        for rel in relationships:
            subject_key = _norm_name(rel.get("subject", ""))
            if not subject_key:
                continue

            person = name_index.get(subject_key)
            if person is None:
                continue

            if "semantic_relationships" not in person:
                person["semantic_relationships"] = []

            sem_rel = _to_semantic_rel(rel, provenance)

            # Deduplicate by (source, target, relation_type)
            exists = any(
                r.get("source")        == sem_rel["source"] and
                r.get("target")        == sem_rel["target"] and
                r.get("relation_type") == sem_rel["relation_type"]
                for r in person["semantic_relationships"]
            )
            if not exists:
                person["semantic_relationships"].append(sem_rel)
                all_ai_rels.append(sem_rel)
                stats["rels_added"] += 1

    # Append new persons at the end
    timeline_persons.extend(new_persons)

    print(
        f"Enrichment complete: {stats['files']} files processed, "
        f"{stats['matched']} persons matched, "
        f"{stats['new']} new persons added, "
        f"{stats['rels_added']} relationships added.",
        flush=True,
    )
    return {
        "people":                   timeline_persons,
        "suggested_people_registry": suggested_people,
        "ai_relationships":          all_ai_rels,
        "suggested_tags_registry":   [],
    }


# ── I/O ───────────────────────────────────────────────────────────────────────

def find_timeline() -> Optional[Path]:
    for p in _TIMELINE_CANDIDATES:
        if p.exists():
            return p
    return None


def load_timeline(path: Path) -> list:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))  # handles optional BOM
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "people" in raw:
        return raw["people"]
    raise ValueError(f"Unexpected timeline format in {path}")


def load_graph_state() -> dict:
    try:
        return json.loads(_GRAPH_STATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"files": {}}


def save_enriched(result: dict, out_path: Path):
    persons = result["people"]
    tmp = out_path.with_suffix(".tmp")
    payload = {
        "people":                    persons,
        "generated_at":              datetime.now(timezone.utc).isoformat(),
        "person_count":              len(persons),
        "suggested_people_registry": result.get("suggested_people_registry", []),
        "ai_relationships":          result.get("ai_relationships", []),
        "suggested_tags_registry":   result.get("suggested_tags_registry", []),
    }
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out_path)
    print(f"Written: {out_path}  ({len(persons):,} persons)", flush=True)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timeline", type=str, default="",
                        help="path to sacred_timeline_current.json (auto-detected if omitted)")
    parser.add_argument("--graph",    type=str, default="",
                        help="path to graph_state.json (default: ./graph_state.json)")
    parser.add_argument("--out",      type=str, default="",
                        help="output path (default: ./sacred_timeline_enriched.json)")
    parser.add_argument("--verbose",  action="store_true",
                        help="print each name match/add")
    parser.add_argument("--dry-run",  action="store_true",
                        help="compute enrichment but do not write output")
    args = parser.parse_args()

    # Resolve paths
    timeline_path = Path(args.timeline) if args.timeline else find_timeline()
    if not timeline_path or not timeline_path.exists():
        print("ERROR: cannot find sacred_timeline_current.json", flush=True)
        sys.exit(1)

    graph_path = Path(args.graph) if args.graph else _GRAPH_STATE
    out_path   = Path(args.out)   if args.out   else _ENRICHED_OUT

    print(f"Timeline : {timeline_path}  ({timeline_path.stat().st_size:,} bytes)", flush=True)
    print(f"Graph    : {graph_path}", flush=True)
    print(f"Output   : {out_path}", flush=True)
    print(flush=True)

    persons     = load_timeline(timeline_path)
    graph_state = json.loads(graph_path.read_text(encoding="utf-8")) if graph_path.exists() else {"files": {}}

    complete = sum(1 for e in graph_state.get("files", {}).values() if e.get("status") == "complete")
    print(f"Timeline persons : {len(persons):,}", flush=True)
    print(f"Graph files      : {len(graph_state.get('files', {})):,} total, {complete} complete", flush=True)
    print(flush=True)

    result = enrich(persons, graph_state, verbose=args.verbose)

    if not args.dry_run:
        save_enriched(result, out_path)
        sp = len(result.get("suggested_people_registry", []))
        ar = len(result.get("ai_relationships", []))
        if sp or ar:
            print(f"Review queue: {sp} suggested people, {ar} AI relationship edges", flush=True)
    else:
        print("(dry-run — output not written)", flush=True)


if __name__ == "__main__":
    main()
