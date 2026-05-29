#!/usr/bin/env python3
"""Parse Genesis_Tag_Vocabulary.txt into a structured JSON file."""
from __future__ import annotations
import json
from pathlib import Path

SRC = Path(__file__).parent / "Genesis_Tag_Vocabulary.txt"
DST = Path(__file__).parent / "Genesis_Tag_Vocabulary.json"

SECTION_HEADERS = {
    "COGNITIVE TAGS":         "cognitive_tags",
    "MATH TAGS":              "math_tags",
    "SCIENTIFIC DOMAIN TAGS": "domain_tags",
    "PIPELINE FLAGS":         "pipeline_flags",
    "SCHOOLS":                "schools",
}
SECTION_LABELS = {
    "cognitive_tags": "Cognitive Tags",
    "math_tags":      "Math Tags",
    "domain_tags":    "Scientific Domain Tags",
    "pipeline_flags": "Pipeline Flags",
    "schools":        "Schools",
}

def parse(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    all_groups: dict[str, list[list[str]]] = {}
    current_section: str | None = None
    current_group: list[str] = []

    for line in lines:
        s = line.strip()
        # Detect section header (e.g. "COGNITIVE TAGS — CLOSED VOCABULARY")
        matched = next((sid for hdr, sid in SECTION_HEADERS.items() if s.startswith(hdr)), None)
        if matched:
            if current_section and current_group:
                all_groups[current_section].append(current_group)
                current_group = []
            current_section = matched
            all_groups.setdefault(current_section, [])
            continue
        if current_section is None:
            continue
        if not s:
            if current_group:
                all_groups[current_section].append(current_group)
                current_group = []
        else:
            current_group.append(s)

    if current_section and current_group:
        all_groups[current_section].append(current_group)

    sections_out: list[dict] = []
    all_tags_map: dict[str, str] = {}

    for sid in SECTION_HEADERS.values():
        groups = all_groups.get(sid, [])
        flat = [tag for g in groups for tag in g]
        for tag in flat:
            all_tags_map[tag] = sid
        sections_out.append({
            "id":              sid,
            "label":           SECTION_LABELS[sid],
            "vocabulary_type": "closed",
            "count":           len(flat),
            "tags":            flat,
            "groups":          groups,
        })

    return {
        "meta": {
            "source":     "Genesis_Tag_Vocabulary.txt",
            "total_tags": len(all_tags_map),
            "sections":   len(sections_out),
        },
        "sections": sections_out,
        "all_tags":  all_tags_map,
    }

if __name__ == "__main__":
    data = parse(SRC)
    DST.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {DST.name}")
    for s in data["sections"]:
        print(f"  {s['label']}: {s['count']} tags in {len(s['groups'])} groups")
    print(f"  Total: {data['meta']['total_tags']} unique tags")
