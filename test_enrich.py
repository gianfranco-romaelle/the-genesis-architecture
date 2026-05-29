#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for enrich.py"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, __import__("pathlib").Path(__file__).parent.__str__())
from enrich import _norm_name, _build_name_index, _find_by_suffix, _to_semantic_rel, enrich

PASS = 0; FAIL = 0

def ok(name):
    global PASS; PASS += 1; print(f"  PASS  {name}", flush=True)

def fail(name, msg=""):
    global FAIL; FAIL += 1; print(f"  FAIL  {name}: {msg}", flush=True)


# ── _norm_name ────────────────────────────────────────────────────────────────
print("--- _norm_name")

pairs = [
    ("Auguste Laurent",    "auguste laurent"),
    ("Évariste Galois",    "evariste galois"),
    ("Augustin-Louis Cauchy", "augustin-louis cauchy"),
    ("  Poincaré  ",       "poincare"),
    ("EULER",              "euler"),
]
for raw, expected in pairs:
    got = _norm_name(raw)
    if got == expected:
        ok(f"norm({raw!r}) == {expected!r}")
    else:
        fail(f"norm({raw!r})", f"got {got!r}, expected {expected!r}")


# ── _build_name_index ─────────────────────────────────────────────────────────
print("--- _build_name_index")

persons = [
    {"name": "Auguste Laurent"},
    {"name": "Évariste Galois"},
    {"name": ""},  # should be dropped
]
idx = _build_name_index(persons)
if "auguste laurent" in idx and "evariste galois" in idx and "" not in idx:
    ok("_build_name_index basic")
else:
    fail("_build_name_index basic", str(list(idx.keys())))


# ── _find_by_suffix ───────────────────────────────────────────────────────────
print("--- _find_by_suffix")

suffix_persons = [
    {"name": "John von Neumann"},
    {"name": "Ada Lovelace"},
    {"name": "John Adams"},
    {"name": "Samuel Adams"},
]
suffix_index = _build_name_index(suffix_persons)

# exact suffix match (2 words)
r = _find_by_suffix("von Neumann", suffix_index)
if r and r["name"] == "John von Neumann":
    ok("_find_by_suffix: 2-word suffix unambiguous match")
else:
    fail("_find_by_suffix: 2-word suffix", str(r))

# single word -> None (too ambiguous)
r2 = _find_by_suffix("Neumann", suffix_index)
if r2 is None:
    ok("_find_by_suffix: single word returns None")
else:
    fail("_find_by_suffix: single word", str(r2))

# multiple matches -> None (ambiguous)
r3 = _find_by_suffix("Adams", suffix_index)
if r3 is None:
    ok("_find_by_suffix: ambiguous single word returns None")
else:
    fail("_find_by_suffix: ambiguous single word", str(r3))

# no match -> None
r4 = _find_by_suffix("Euler Lagrange", suffix_index)
if r4 is None:
    ok("_find_by_suffix: no match returns None")
else:
    fail("_find_by_suffix: no match", str(r4))

# suffix match with accents stripped
accent_persons = [{"name": "Henri Poincare"}]
accent_index = _build_name_index(accent_persons)
r5 = _find_by_suffix("Poincare", accent_index)
if r5 is None:
    ok("_find_by_suffix: single word still None even after norm")
else:
    fail("_find_by_suffix: single word accent", str(r5))

# 2-word accent suffix
r6 = _find_by_suffix("Henri Poincare", accent_index)
if r6 and r6["name"] == "Henri Poincare":
    ok("_find_by_suffix: exact 2-word match via norm")
else:
    fail("_find_by_suffix: exact 2-word", str(r6))


# ── _to_semantic_rel ──────────────────────────────────────────────────────────
print("--- _to_semantic_rel")

rel = {
    "subject": "Pons", "predicate": "collaborated with",
    "object": "Fleischmann", "evidence": "in_source_text_direct",
}
sr = _to_semantic_rel(rel, "graph_builder:cold_fusion.pdf")
if (sr["source"] == "Pons" and sr["target"] == "Fleischmann"
        and sr["relation_type"] == "collaborated with"
        and sr["evidence_type"] == "in_source_text_direct"
        and sr["assertion_layer"] == "ai_hypothesis"
        and "cold_fusion.pdf" in sr["provenance"]):
    ok("_to_semantic_rel mapping correct")
else:
    fail("_to_semantic_rel", str(sr))


# ── enrich: match existing person ────────────────────────────────────────────
print("--- enrich: match")

timeline = [
    {"name": "Auguste Laurent", "birth_year": 1807, "death_year": 1853, "calculus_number": 4},
    {"name": "Évariste Galois",  "birth_year": 1811, "death_year": 1832, "calculus_number": 4},
]
graph_state = {"files": {"chemistry.pdf": {
    "status": "complete",
    "persons": [{"name": "Auguste Laurent", "birth_year": 1807,
                 "calculus_number": 4, "evidence": "in_source_text_direct"}],
    "relationships": [{"subject": "Auguste Laurent", "predicate": "studied under",
                       "object": "Dumas", "evidence": "in_source_text_direct"}],
}}}

result = enrich(timeline, graph_state)
people = result["people"]

# timeline still has 2 persons (Laurent was matched, not added)
if len(people) == 2:
    ok("enrich: no duplicate added for matched person")
else:
    fail("enrich: duplicate check", f"got {len(people)} persons")

# Laurent should have source_files
laurent = next((p for p in people if _norm_name(p["name"]) == "auguste laurent"), None)
if laurent and "chemistry.pdf" in laurent.get("source_files", []):
    ok("enrich: source_files added to matched person")
else:
    fail("enrich: source_files", str(laurent))

# Relationship added to Laurent
rels = laurent.get("semantic_relationships", []) if laurent else []
if rels and rels[0]["relation_type"] == "studied under" and rels[0]["target"] == "Dumas":
    ok("enrich: relationship added to matched person")
else:
    fail("enrich: relationship", str(rels))

# Galois has no relationships
galois = next((p for p in people if _norm_name(p["name"]) == "evariste galois"), None)
if galois and not galois.get("semantic_relationships"):
    ok("enrich: unrelated person untouched")
else:
    fail("enrich: unrelated person", str(galois))

# matched person → NOT in suggested_people_registry
if not result["suggested_people_registry"]:
    ok("enrich: matched person not in suggested_people_registry")
else:
    fail("enrich: suggested_people_registry should be empty for matched", str(result["suggested_people_registry"]))


# ── enrich: add new person ────────────────────────────────────────────────────
print("--- enrich: new person")

timeline2 = [{"name": "Cauchy", "birth_year": 1789, "calculus_number": 4}]
graph_state2 = {"files": {"paper.pdf": {
    "status": "complete",
    "persons": [
        {"name": "Unknown NewPerson", "birth_year": 1950,
         "calculus_number": 6, "nationality": "American",
         "occupation": "physicist", "evidence": "in_source_text_direct"},
    ],
    "relationships": [],
}}}
result2 = enrich(timeline2, graph_state2)
people2 = result2["people"]

if len(people2) == 2:
    ok("enrich: new person appended")
else:
    fail("enrich: new person append", f"len={len(people2)}")

new_p = next((p for p in people2 if "NewPerson" in p.get("name", "")), None)
if new_p and new_p.get("country") == "American" and new_p.get("field") == "physicist":
    ok("enrich: new person fields mapped correctly")
else:
    fail("enrich: new person fields", str(new_p))

# unmatched person appears in suggested_people_registry
sp2 = result2["suggested_people_registry"]
if len(sp2) == 1 and sp2[0]["name"] == "Unknown NewPerson" and sp2[0]["source_title"] == "paper.pdf":
    ok("enrich: new person in suggested_people_registry")
else:
    fail("enrich: suggested_people_registry for new person", str(sp2))


# ── enrich: skip non-complete files ──────────────────────────────────────────
print("--- enrich: status filtering")

timeline3 = [{"name": "Euler", "birth_year": 1707, "calculus_number": 3}]
graph_state3 = {"files": {
    "done.pdf":  {"status": "complete",  "persons": [{"name": "Lagrange", "birth_year": 1736, "calculus_number": 3}], "relationships": []},
    "error.pdf": {"status": "error",     "persons": [{"name": "BadPerson", "birth_year": 1800}], "relationships": []},
    "pend.pdf":  {"status": "pending",   "persons": [{"name": "AlsoBad",  "birth_year": 1800}], "relationships": []},
}}
result3 = enrich(timeline3, graph_state3)
names3 = [p["name"] for p in result3["people"]]
if "Lagrange" in names3 and "BadPerson" not in names3 and "AlsoBad" not in names3:
    ok("enrich: only complete files processed")
else:
    fail("enrich: status filtering", str(names3))


# ── enrich: suffix-match merges into existing person ─────────────────────────
print("--- enrich: suffix match")

timeline_s = [
    {"name": "John von Neumann", "birth_year": 1903, "calculus_number": 6},
]
graph_state_s = {"files": {"quantum.pdf": {
    "status": "complete",
    "persons": [{"name": "von Neumann", "birth_year": 1903, "calculus_number": 6,
                 "evidence": "in_source_text_direct"}],
    "relationships": [{"subject": "von Neumann", "predicate": "formalized",
                       "object": "quantum mechanics", "evidence": "in_source_text_direct"}],
}}}
result_s = enrich(timeline_s, graph_state_s)
people_s = result_s["people"]

# should NOT add a new person — von Neumann matched via suffix
if len(people_s) == 1:
    ok("enrich suffix: no duplicate created")
else:
    fail("enrich suffix: duplicate check", f"got {len(people_s)} persons")

# relationship must land on John von Neumann
jvn = next((p for p in people_s if "von Neumann" in p["name"]), None)
rels_s = jvn.get("semantic_relationships", []) if jvn else []
if rels_s and rels_s[0]["relation_type"] == "formalized":
    ok("enrich suffix: relationship attached to matched person")
else:
    fail("enrich suffix: relationship", str(rels_s))

# source_files added
if jvn and "quantum.pdf" in jvn.get("source_files", []):
    ok("enrich suffix: source_files added to suffix-matched person")
else:
    fail("enrich suffix: source_files", str(jvn))

# suffix-matched person NOT in suggested_people_registry
if not result_s["suggested_people_registry"]:
    ok("enrich suffix: suffix-matched person not in suggested_people_registry")
else:
    fail("enrich suffix: suggested_people_registry should be empty", str(result_s["suggested_people_registry"]))

# ai_relationships populated
if result_s["ai_relationships"] and result_s["ai_relationships"][0]["relation_type"] == "formalized":
    ok("enrich suffix: ai_relationships populated")
else:
    fail("enrich suffix: ai_relationships", str(result_s["ai_relationships"]))


# ── enrich: relationship deduplication ───────────────────────────────────────
print("--- enrich: deduplication")

timeline4 = [{"name": "Pons", "birth_year": 1943, "calculus_number": 6}]
rel_entry = {"subject": "Pons", "predicate": "collaborated with",
             "object": "Fleischmann", "evidence": "in_source_text_direct"}
graph_state4 = {"files": {
    "file_a.pdf": {"status": "complete", "persons": [{"name": "Pons", "birth_year": 1943, "calculus_number": 6, "evidence": "in_source_text_direct"}], "relationships": [rel_entry]},
    "file_b.pdf": {"status": "complete", "persons": [{"name": "Pons", "birth_year": 1943, "calculus_number": 6, "evidence": "in_source_text_direct"}], "relationships": [rel_entry]},
}}
result4 = enrich(timeline4, graph_state4)
people4 = result4["people"]

pons = next((p for p in people4 if p["name"] == "Pons"), None)
rels4 = pons.get("semantic_relationships", []) if pons else []
if len(rels4) == 1:
    ok("enrich: duplicate relationship deduplicated")
else:
    fail("enrich: dedup", f"got {len(rels4)} rels, expected 1")

# ai_relationships also deduplicated (both files have same rel → 1 in list)
if len(result4["ai_relationships"]) == 1:
    ok("enrich: ai_relationships also deduplicated")
else:
    fail("enrich: ai_relationships dedup", f"got {len(result4['ai_relationships'])}")


# ── summary ───────────────────────────────────────────────────────────────────
print()
print(f"Results: {PASS} passed, {FAIL} failed")
if __name__ == "__main__":
    sys.exit(FAIL)
