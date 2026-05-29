#!/usr/bin/env python3
"""
normalize.py — run after each gemini_pipeline.py session.

Reads:
  - sacred_timeline_current.json (authoritative people roster)
  - w4_entities_enriched.jsonl   (tags, centrality, cluster from W4)
  - w4_relationships.jsonl       (semantic edges from W4)
  - w3_relationships.jsonl       (topology edges from W3)
  - w5_bibliography.jsonl        (bibliography from W5)
  - w2_queue_entities.jsonl      (newly discovered entities from W2)

Writes:
  - sacred_timeline_enriched.json                           (constellation view)
  - ../sacred-timeline/public/generated/
      historical-entity-graph.snapshot.json                 (live app auto-reload)
      drive_index_WIP.json                                  (W2 index records)
      drive_semantic_embeddings_WIP.json                    (W3 embeddings)
  - dedup_conflicts.json
  - suggested_people.jsonl

All pipeline-generated relationships carry assertion_layer: "ai_hypothesis".
"""

from __future__ import annotations

import datetime
import json
import re
import uuid
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
SACRED_TIMELINE_DIR = BASE_DIR.parent / "sacred-timeline"
GENERATED_DIR = SACRED_TIMELINE_DIR / "public" / "generated"


def _find(*candidates) -> Path | None:
    for p in candidates:
        path = Path(p)
        if path.exists():
            return path
    return None


TIMELINE_JSON = _find(
    SACRED_TIMELINE_DIR / "public" / "sacred_timeline_current.json",
    SACRED_TIMELINE_DIR / "sacred_timeline_current.json",
    BASE_DIR / "sacred_timeline_5-10-2026.json",
)
CALCULI_EXPORT = _find(
    SACRED_TIMELINE_DIR / "Calculi_0-2_Export_5-10-2026.txt",
    BASE_DIR / "Calculi_0-2_Export_5-10-2026.txt",
)

# JSONL inputs
W4_ENRICHED = BASE_DIR / "w4_entities_enriched.jsonl"
W4_RELS = BASE_DIR / "w4_relationships.jsonl"
W3_RELS = BASE_DIR / "w3_relationships.jsonl"
W5_BIB = BASE_DIR / "w5_bibliography.jsonl"
W2_INDEX = BASE_DIR / "w2_index_records.jsonl"
W2_QUEUE = BASE_DIR / "w2_queue_entities.jsonl"
W3_EMBEDDINGS = BASE_DIR / "w3_embeddings.jsonl"

# Output files
ENRICHED_OUT = BASE_DIR / "sacred_timeline_enriched.json"
SNAPSHOT_OUT = GENERATED_DIR / "historical-entity-graph.snapshot.json"
DRIVE_INDEX_OUT = GENERATED_DIR / "drive_index_WIP.json"
DRIVE_SEMANTIC_OUT = GENERATED_DIR / "drive_semantic_embeddings_WIP.json"
DEDUP_CONFLICTS_OUT = BASE_DIR / "dedup_conflicts.json"
SUGGESTED_PEOPLE_OUT = BASE_DIR / "suggested_people.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_person_id(name: str, birth_year) -> str:
    slug = re.sub(r"[^\w\s]", "", name.lower()).strip().replace(" ", "_")
    return f"{slug}_{birth_year}" if birth_year is not None else f"{slug}_unknown"


def norm(name: str) -> str:
    return name.lower().strip()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                except json.JSONDecodeError:
                    pass
    return records


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def _lifespan_quality(rec: dict) -> int:
    val = rec.get("lifespan_raw", "")
    return 1 if val and val != "nan" else 0


def _merge_pair(base: dict, other: dict) -> dict:
    merged = dict(base)
    if _lifespan_quality(other) > _lifespan_quality(merged):
        merged["lifespan_raw"] = other["lifespan_raw"]
    if merged.get("portrait_url") is None and other.get("portrait_url") == "Picture":
        merged["portrait_url"] = "Picture"
    if "date_note" not in merged and "date_note" in other:
        merged["date_note"] = other["date_note"]
    return merged


def dedup_timeline(records: list[dict]) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple, list] = defaultdict(list)
    for rec in records:
        key = (norm(rec.get("name", "")), rec.get("birth_year"), rec.get("calculus_number"))
        groups[key].append(rec)

    merged_list: list[dict] = []
    conflicts: list[dict] = []

    for key, group in groups.items():
        if len(group) == 1:
            merged_list.append(group[0])
            continue
        if all(r == group[0] for r in group[1:]):
            merged_list.append(group[0])
            continue

        death_years = {r.get("death_year") for r in group if r.get("death_year") is not None}
        if len(death_years) > 1:
            conflicts.append({"key": list(key), "records": group})

        best = group[0]
        for other in group[1:]:
            best = _merge_pair(best, other)

        canonical_name = max(
            (r.get("name", "") for r in group),
            key=lambda n: (n != n.lower(), len(n)),
        )
        best = {**best, "name": canonical_name}
        merged_list.append(best)

    return merged_list, conflicts


# ---------------------------------------------------------------------------
# Pipeline flags
# ---------------------------------------------------------------------------
ENRICHMENT_TAG_FIELDS = [
    "schools", "suggested_schools",
    "cognitive_tags", "suggested_cognitive_tags",
    "math_tags", "suggested_math_tags",
    "domain_tags", "suggested_domain_tags",
    "suggested_pipeline_flags",
]


def compute_pipeline_flags(person: dict) -> list[str]:
    flags: list[str] = []
    if person.get("calculus_number") is not None:
        flags.append("Era-Assigned")
    has_tags = any(person.get(f) for f in ["schools", "cognitive_tags", "math_tags", "domain_tags"])
    if has_tags:
        flags.append("Auto-Tagged")
    rels = person.get("semantic_relationships", [])
    if rels:
        flags.append("Edges-Linked")
        flags.append("Graph-Ready" if has_tags else "Graph-Partial")
    else:
        flags.append("Relationship-Incomplete")
    return flags


# ---------------------------------------------------------------------------
# People-tag resolution
# ---------------------------------------------------------------------------
def resolve_people_tags(
    people_tags: list, roster_by_norm: dict
) -> tuple[list, list]:
    resolved, unresolved = [], []
    for tag in people_tags:
        name = tag.get("name", "")
        matches = roster_by_norm.get(norm(name), [])
        if matches:
            person = matches[0]
            resolved.append({
                "person_id": person["_pid"],
                "name": person["name"],
                "relation": tag.get("relation", "mentioned"),
            })
        else:
            unresolved.append(tag)
    return resolved, unresolved


# ---------------------------------------------------------------------------
# Historical entity graph snapshot builder
# ---------------------------------------------------------------------------
DOMAIN_HISTORY = "domain_history"
DOMAIN_SCIENCE = "domain_science"
DOMAIN_PHILOSOPHY = "domain_philosophy"


def _slug(name: str, birth_year) -> str:
    clean = re.sub(r"[^\w\s]", "", name.lower()).strip().replace(" ", "-")
    year_part = str(birth_year) if birth_year is not None else "unknown"
    return f"{clean}-{year_part}"


def _infer_domain_ids(person: dict) -> list[str]:
    field = (person.get("field") or "").lower()
    domains = [DOMAIN_HISTORY]
    if any(w in field for w in ("math", "physics", "chem", "bio", "astro", "science")):
        domains.append(DOMAIN_SCIENCE)
    if any(w in field for w in ("philos", "logic", "ethics", "metaphys")):
        domains.append(DOMAIN_PHILOSOPHY)
    return domains


def _make_resolved_field(value, source_record_id: str,
                          source_kind: str = "sacred_timeline_enriched",
                          confidence: float = 0.7) -> dict:
    return {
        "mergedValue": value,
        "effectiveValue": value,
        "precedence": "merged",
        "values": [{
            "field": "displayName",  # overwritten per field
            "value": value,
            "sourceRecordId": source_record_id,
            "sourceKind": source_kind,
            "confidence": confidence,
        }],
        "provenance": [],
    }


def build_snapshot(
    people: list[dict],
    generated_at: str,
    curated_master_path: str,
) -> dict:
    source_records = []
    canonical_entities = []
    candidates = []

    for i, person in enumerate(people):
        name = person["name"]
        birth = person.get("birth_year")
        death = person.get("death_year")
        pid = person.get("person_id") or make_person_id(name, birth)
        sl = _slug(name, birth)
        src_id = f"source_record_curated_{sl}"
        entity_id = f"person_{sl}"
        domain_ids = _infer_domain_ids(person)

        # Source record
        sr: dict = {
            "id": src_id,
            "kind": "sacred_timeline_enriched",
            "recordType": "curated_person",
            "externalId": str(i),
            "title": name,
            "displayName": name,
            "slug": sl,
            "normalizedDisplayName": norm(name),
            "sourcePath": f"sacred_timeline_enriched.json#row[{i}]",
            "birthYear": birth,
            "deathYear": death,
            "aliases": [],
            "domainHints": domain_ids,
            "traditionHints": person.get("schools", [])[:3],
            "placeHints": [person["country"]] if person.get("country") else [],
            "rawFields": {
                "index": i,
                "name": name,
                "birth_year": birth,
                "death_year": death,
                "field": person.get("field"),
                "country": person.get("country"),
                "calculus_name": person.get("calculus_name"),
                "calculus_number": person.get("calculus_number"),
                "person_id": pid,
            },
            "fieldValues": [
                {
                    "field": "displayName",
                    "value": name,
                    "sourceKind": "sacred_timeline_enriched",
                    "sourceRecordId": src_id,
                    "confidence": 0.9,
                    "rawPath": "name",
                },
                {
                    "field": "birthYear",
                    "value": birth,
                    "sourceKind": "sacred_timeline_enriched",
                    "sourceRecordId": src_id,
                    "confidence": 0.9,
                    "rawPath": "birth_year",
                },
                {
                    "field": "deathYear",
                    "value": death,
                    "sourceKind": "sacred_timeline_enriched",
                    "sourceRecordId": src_id,
                    "confidence": 0.9,
                    "rawPath": "death_year",
                },
                {
                    "field": "domainIds",
                    "value": domain_ids,
                    "sourceKind": "sacred_timeline_enriched",
                    "sourceRecordId": src_id,
                    "confidence": 0.4,
                    "rawPath": "field",
                },
            ],
            "portraitCandidates": [],
            "confidence": 0.8,
        }
        # Add wikipedia fields if present
        if person.get("wikipedia_url"):
            sr["wikipediaUrl"] = person["wikipedia_url"]
            sr["wikidataId"] = person.get("wikipedia_wikidata_id")
            sr["articlePath"] = person.get("wikipedia_article_text_path")
            sr["citationsPath"] = person.get("wikipedia_citations_path")
            portrait_path = person.get("portrait_url")
            if portrait_path and portrait_path not in ("Picture", None, ""):
                portrait_id = f"source_media_curated_{sl}"
                sr["portraitCandidates"] = [{
                    "id": portrait_id,
                    "sourceRecordId": src_id,
                    "label": f"{name} portrait",
                    "kind": "portrait",
                    "publicAssetPath": portrait_path,
                    "isPreferred": True,
                }]
                sr["fieldValues"].append({
                    "field": "preferredPortrait",
                    "value": portrait_id,
                    "sourceKind": "sacred_timeline_enriched",
                    "sourceRecordId": src_id,
                    "confidence": 0.8,
                    "rawPath": "portrait_url",
                })
        source_records.append(sr)

        # Canonical entity
        ce: dict = {
            "id": entity_id,
            "entityType": "person",
            "slug": sl,
            "mergeStatus": "singleton",
            "reviewStatus": "draft",
            "visibility": "draft",
            "mergeConfidence": 0.8,
            "mergeSignals": [],
            "fieldConflicts": [],
            "sourceRecordIds": [src_id],
            "relatedTextLabels": [],
            "relatedPlaceLabels": [],
            "portraitCandidates": sr.get("portraitCandidates", []),
            "rawCandidateIds": [],
            "updatedAt": generated_at,
            "displayName": {
                "mergedValue": name,
                "effectiveValue": name,
                "precedence": "merged",
                "values": [{"field": "displayName", "value": name, "sourceRecordId": src_id,
                             "sourceKind": "sacred_timeline_enriched", "confidence": 0.9}],
                "provenance": [],
            },
            "aliases": {
                "mergedValue": [],
                "effectiveValue": [],
                "precedence": "merged",
                "values": [],
                "provenance": [],
            },
            "birthYear": {
                "mergedValue": birth,
                "effectiveValue": birth,
                "precedence": "merged",
                "values": [{"field": "birthYear", "value": birth, "sourceRecordId": src_id,
                             "sourceKind": "sacred_timeline_enriched", "confidence": 0.9}],
                "provenance": [],
            },
            "deathYear": {
                "mergedValue": death,
                "effectiveValue": death,
                "precedence": "merged",
                "values": [{"field": "deathYear", "value": death, "sourceRecordId": src_id,
                             "sourceKind": "sacred_timeline_enriched", "confidence": 0.9}],
                "provenance": [],
            },
            "summary": {
                "mergedValue": person.get("wikipedia_description") or f"{name}, {person.get('calculus_name') or 'historical figure'}.",
                "effectiveValue": person.get("wikipedia_description"),
                "precedence": "merged",
                "values": [],
                "provenance": [],
            },
            "domainIds": {
                "mergedValue": domain_ids,
                "effectiveValue": domain_ids,
                "precedence": "merged",
                "values": [{"field": "domainIds", "value": domain_ids, "sourceRecordId": src_id,
                             "sourceKind": "sacred_timeline_enriched", "confidence": 0.4}],
                "provenance": [],
            },
            "traditionIds": {
                "mergedValue": [],
                "effectiveValue": [],
                "precedence": "merged",
                "values": [],
                "provenance": [],
            },
            "tagIds": {
                "mergedValue": [],
                "effectiveValue": [],
                "precedence": "merged",
                "values": [],
                "provenance": [],
            },
            "preferredPortrait": {
                "mergedValue": None,
                "effectiveValue": None,
                "precedence": "merged",
                "values": [],
                "provenance": [],
            },
            "metadata": {
                "calculus_number": person.get("calculus_number"),
                "calculus_name": person.get("calculus_name"),
                "person_id": pid,
                "schools": person.get("schools", []),
                "cognitive_tags": person.get("cognitive_tags", []),
                "math_tags": person.get("math_tags", []),
                "domain_tags": person.get("domain_tags", []),
                "centrality_score": person.get("centrality_score"),
                "cluster_id": person.get("cluster_id"),
                "pipeline_flags": person.get("pipeline_flags", []),
            },
        }

        # Attach portrait if available
        portrait_id = sr.get("fieldValues", [{}])[-1].get("value") if sr.get("portraitCandidates") else None
        if portrait_id:
            ce["preferredPortrait"]["mergedValue"] = portrait_id

        canonical_entities.append(ce)

        # Candidate (minimal)
        candidates.append({
            "id": f"candidate_{sl}",
            "entityType": "person",
            "displayName": name,
            "normalizedName": norm(name),
            "aliases": [],
            "normalizedAliases": [],
            "nameTokens": norm(name).split(),
            "birthYear": birth,
            "deathYear": death,
            "domainIds": domain_ids,
            "traditionIds": [],
            "tagIds": [],
            "relatedTextLabels": [],
            "relatedPlaceLabels": [],
            "sourceRecordIds": [src_id],
            "portraitCandidates": sr.get("portraitCandidates", []),
            "confidence": 0.8,
            "mergeSignals": [],
        })

    public_count = sum(
        1 for p in people if p.get("portrait_url") not in (None, "Picture", "")
    )

    snapshot = {
        "generatedAt": generated_at,
        "sourcePaths": {
            "curatedMasterPath": str(ENRICHED_OUT),
            "wikipediaJobsPath": str(SACRED_TIMELINE_DIR / "public" / "wikipedia_assistant"),
            "wikipediaRecordsPath": str(SACRED_TIMELINE_DIR / "public" / "wikipedia_assistant" / "records"),
            "imageOutputDir": str(GENERATED_DIR / "wikipedia_images"),
        },
        "summary": {
            "curatedRecordCount": len(people),
            "wikipediaRecordCount": 0,
            "candidateCount": len(candidates),
            "canonicalEntityCount": len(canonical_entities),
            "publicEntityCount": public_count,
            "draftEntityCount": len(canonical_entities) - public_count,
        },
        "sourceRecords": source_records,
        "candidates": candidates,
        "mergeDecisions": [],
        "canonicalEntities": canonical_entities,
    }
    return snapshot


# ---------------------------------------------------------------------------
# Drive index / semantic payloads from W2/W3 output
# ---------------------------------------------------------------------------
def build_drive_index(w2_records: list[dict]) -> dict:
    index_records = [r for r in w2_records if r.get("record_type") == "index"]
    queue_entities = [r for r in w2_records if r.get("record_type") == "queue_entity"]
    proposed = [r for r in w2_records if r.get("record_type") == "proposed_addition"]
    return {
        "generatedAt": datetime.datetime.utcnow().isoformat() + "Z",
        "indexRecordCount": len(index_records),
        "queueEntityCount": len(queue_entities),
        "proposedAdditionCount": len(proposed),
        "indexRecords": index_records,
        "queueEntities": queue_entities,
        "proposedAdditions": proposed,
    }


def build_drive_semantic(w3_records: list[dict]) -> dict:
    embeddings = [r for r in w3_records if r.get("record_type") == "embedding"]
    return {
        "generatedAt": datetime.datetime.utcnow().isoformat() + "Z",
        "embeddingCount": len(embeddings),
        "embeddings": embeddings,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not TIMELINE_JSON:
        print("ERROR: sacred_timeline_current.json not found")
        return

    print(f"Loading timeline from {TIMELINE_JSON} ...")
    with open(TIMELINE_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    print(f"  {len(raw)} raw records")

    # --- Dedup ---
    people, conflicts = dedup_timeline(raw)
    removed = len(raw) - len(people)
    print(f"  After dedup: {len(people)} unique ({removed} removed, {len(conflicts)} conflicts flagged)")

    # Assign internal PIDs
    for p in people:
        p["_pid"] = make_person_id(p["name"], p.get("birth_year"))

    # Roster lookup
    roster_by_norm: dict[str, list] = defaultdict(list)
    for p in people:
        roster_by_norm[norm(p["name"])].append(p)

    # --- Load enrichment ---
    print("Loading JSONL files ...")
    w4_enriched = read_jsonl(W4_ENRICHED)
    w4_rels = read_jsonl(W4_RELS)
    w3_rels = read_jsonl(W3_RELS)
    w5_bibs = read_jsonl(W5_BIB)
    w2_records = read_jsonl(W2_INDEX) + read_jsonl(W2_QUEUE)
    w3_embeddings = read_jsonl(W3_EMBEDDINGS)
    print(
        f"  w4_enriched={len(w4_enriched)}  w4_rels={len(w4_rels)}  "
        f"w3_rels={len(w3_rels)}  w5_bibs={len(w5_bibs)}"
    )

    # Index enrichment
    enriched_by_norm: dict[str, dict] = {}
    for rec in w4_enriched:
        enriched_by_norm[norm(rec.get("name", ""))] = rec

    # Index relationships (W4 + W3 combined), enforce assertion_layer
    calculus_by_norm = {norm(p["name"]): p.get("calculus_number") for p in people}
    rels_by_source: dict[str, list] = defaultdict(list)
    for rel in w4_rels + w3_rels:
        rel.setdefault("assertion_layer", "ai_hypothesis")
        if rel.get("calculus_target") is None:
            rel["calculus_target"] = calculus_by_norm.get(norm(rel.get("target", "")))
        rels_by_source[norm(rel.get("source", ""))].append(rel)

    # Index bibliography
    bib_by_person: dict[str, list] = defaultdict(list)
    for bib in w5_bibs:
        bib.setdefault("assertion_layer", "ai_hypothesis")
        sp = norm(bib.get("source_person", ""))
        if sp:
            bib_by_person[sp].append(bib)

    # --- Merge ---
    print("Merging enrichment ...")
    suggested_people_accumulator: list[dict] = []

    for person in people:
        key = norm(person["name"])
        enrichment = enriched_by_norm.get(key, {})

        for field in ENRICHMENT_TAG_FIELDS:
            if field in enrichment:
                person[field] = enrichment[field]
        for field in ("centrality_score", "cluster_id"):
            if field in enrichment:
                person[field] = enrichment[field]

        person["person_id"] = enrichment.get("person_id") or person.pop("_pid")
        person.pop("_pid", None)

        person["semantic_relationships"] = [
            {
                "source": r.get("source"),
                "target": r.get("target"),
                "relation_type": r.get("relation_type"),
                "weight": r.get("weight"),
                "confidence": r.get("confidence"),
                "evidence_type": r.get("evidence_type"),
                "provenance": r.get("provenance"),
                "assertion_layer": "ai_hypothesis",
            }
            for r in rels_by_source.get(key, [])
        ]

        resolved_bibs = []
        for bib in bib_by_person.get(key, []):
            resolved_tags, unresolved = resolve_people_tags(
                bib.get("people_tags", []), roster_by_norm
            )
            bib = {**bib, "people_tags": resolved_tags}
            for utag in unresolved:
                suggested_people_accumulator.append({
                    "bib_title": bib.get("title"),
                    "source_person": person["name"],
                    **utag,
                })
            for stag in bib.get("suggested_people_tags", []):
                suggested_people_accumulator.append({
                    "bib_title": bib.get("title"),
                    "source_person": person["name"],
                    **stag,
                })
            resolved_bibs.append(bib)
        person["associated_texts"] = resolved_bibs

        person["pipeline_flags"] = compute_pipeline_flags(person)

    # --- Optional Calculi export ---
    sheaf_package: dict = {}
    mathematical_kernel_bundle: dict = {}
    if CALCULI_EXPORT:
        try:
            calculi_data = json.loads(CALCULI_EXPORT.read_text(encoding="utf-8"))
            sheaf_package = calculi_data.get("sheaf_package", {})
            mathematical_kernel_bundle = calculi_data.get("mathematical_kernel_bundle", {})
            bundles_by_norm = {
                norm(b.get("name", "")): b
                for b in calculi_data.get("entity_bundles", [])
            }
            for person in people:
                bundle = bundles_by_norm.get(norm(person["name"]), {})
                for field in ("pedagogical_significance", "image_status", "confidence_level"):
                    if field in bundle and field not in person:
                        person[field] = bundle[field]
        except Exception as exc:
            print(f"  Warning: Calculi export skipped: {exc}")

    # --- Suggested registries ---
    reg: dict[str, set] = {
        "suggested_math_tags": set(),
        "suggested_cognitive_tags": set(),
        "suggested_schools": set(),
        "suggested_domain_tags": set(),
    }
    for p in people:
        for field in reg:
            for tag in p.get(field, []):
                if tag:
                    reg[field].add(tag)
    suggested_tags_registry = {k: sorted(v) for k, v in reg.items()}

    seen_suggested: set[str] = set()
    suggested_people_registry: list[dict] = []
    for entry in suggested_people_accumulator:
        key_str = norm(entry.get("name", ""))
        if key_str and key_str not in seen_suggested:
            seen_suggested.add(key_str)
            suggested_people_registry.append(entry)

    all_rels = w4_rels + w3_rels
    influence_graphs = [
        {
            "source": r.get("source"),
            "target": r.get("target"),
            "relation_type": r.get("relation_type"),
            "weight": r.get("weight"),
            "confidence": r.get("confidence"),
            "assertion_layer": "ai_hypothesis",
        }
        for r in all_rels
    ]

    # --- Write sacred_timeline_enriched.json ---
    enriched_output = {
        "people": people,
        "sheaf_package": sheaf_package,
        "school_influence_bundle": {"influence_graphs": influence_graphs},
        "mathematical_kernel_bundle": mathematical_kernel_bundle,
        "suggested_tags_registry": suggested_tags_registry,
        "suggested_people_registry": suggested_people_registry,
    }
    print(f"Writing {ENRICHED_OUT} ...")
    with open(ENRICHED_OUT, "w", encoding="utf-8") as f:
        json.dump(enriched_output, f, ensure_ascii=False, indent=2)

    edge_count = sum(len(p.get("semantic_relationships", [])) for p in people)
    tagged = sum(1 for p in people if "Auto-Tagged" in p.get("pipeline_flags", []))
    graph_ready = sum(1 for p in people if "Graph-Ready" in p.get("pipeline_flags", []))
    print(f"  {len(people)} people  {edge_count} edges  {tagged} tagged  {graph_ready} graph-ready")

    # --- Write historical-entity-graph.snapshot.json ---
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.datetime.utcnow().isoformat() + "Z"
    snapshot = build_snapshot(people, generated_at, str(TIMELINE_JSON))
    print(f"Writing {SNAPSHOT_OUT} ...")
    with open(SNAPSHOT_OUT, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    print(f"  {snapshot['summary']['canonicalEntityCount']} canonical entities")

    # --- Write drive_index_WIP.json ---
    drive_index = build_drive_index(w2_records)
    print(f"Writing {DRIVE_INDEX_OUT} ...")
    with open(DRIVE_INDEX_OUT, "w", encoding="utf-8") as f:
        json.dump(drive_index, f, ensure_ascii=False, indent=2)

    # --- Write drive_semantic_embeddings_WIP.json ---
    drive_semantic = build_drive_semantic(w3_embeddings)
    print(f"Writing {DRIVE_SEMANTIC_OUT} ...")
    with open(DRIVE_SEMANTIC_OUT, "w", encoding="utf-8") as f:
        json.dump(drive_semantic, f, ensure_ascii=False, indent=2)

    # --- Auxiliary outputs ---
    if conflicts:
        with open(DEDUP_CONFLICTS_OUT, "w", encoding="utf-8") as f:
            json.dump(conflicts, f, ensure_ascii=False, indent=2)
        print(f"  {len(conflicts)} dedup conflicts → {DEDUP_CONFLICTS_OUT}")

    if suggested_people_accumulator:
        with open(SUGGESTED_PEOPLE_OUT, "w", encoding="utf-8") as f:
            for entry in suggested_people_accumulator:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"  {len(suggested_people_accumulator)} unresolved people → {SUGGESTED_PEOPLE_OUT}")

    print("Done.")


if __name__ == "__main__":
    main()
