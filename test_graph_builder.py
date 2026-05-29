#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for graph_builder.py"""
import sys, os, json, pathlib, tempfile, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import graph_builder as gb
from graph_builder import (
    _parse_response, _dedup_persons, _fix_calculus_numbers,
    _filter_person_person_rels,
    era_distribution, save_graph_state, get_library_root, _get_pending,
)

PASS = 0; FAIL = 0

def ok(name):
    global PASS; PASS += 1; print(f"  PASS  {name}", flush=True)

def fail(name, msg=""):
    global FAIL; FAIL += 1; print(f"  FAIL  {name}: {msg}", flush=True)

# ── _dedup_persons ────────────────────────────────────────────────────────────
print("--- _dedup_persons")

dupes = [
    {"name": "Auguste Laurent", "calculus_number": 4},
    {"name": "Auguste Laurent", "calculus_number": 4},  # exact
    {"name": "auguste laurent", "calculus_number": 4},  # case variant
    {"name": "Galois",          "calculus_number": 4},
]
r = _dedup_persons(dupes)
expected = [{"name": "Auguste Laurent", "calculus_number": 4}, {"name": "Galois", "calculus_number": 4}]
if r == expected:
    ok("dedup basic")
else:
    fail("dedup basic", str(r))

r2 = _dedup_persons([{"name": ""}, {"name": "Alice"}, {"name": ""}])
if len(r2) == 1 and r2[0]["name"] == "Alice":
    ok("dedup empty names dropped")
else:
    fail("dedup empty names dropped", str(r2))

if _dedup_persons([]) == []:
    ok("dedup empty list")
else:
    fail("dedup empty list")

# ── _fix_calculus_numbers ─────────────────────────────────────────────────────
print("--- _fix_calculus_numbers")

cases = [
    # (birth_year, input_cn, expected_cn, label)
    (1942, 4, 6, "born 1942 -> Calc6"),
    (1930, 5, 6, "born 1930 -> Calc6"),
    (1929, 5, 5, "born 1929 -> Calc5 ok (already correct)"),
    (1880, 3, 5, "born 1880 -> Calc5 min"),
    (1850, 4, 5, "born 1850 -> Calc5 min"),
    (1849, 4, 4, "born 1849 -> Calc4 ok"),
    (1810, 2, 4, "born 1810 -> Calc4 min"),
    (1780, 3, 4, "born 1780 -> Calc4 min"),
    (1779, 3, 3, "born 1779 -> Calc3 ok"),
    (1789, 3, 4, "born 1789 (Cauchy) -> Calc4 min"),
    (None, 4, 4, "no birth year -> unchanged"),
    (1600, 1, 1, "Calc1 era -> unchanged"),
]
for by, input_cn, expected_cn, label in cases:
    persons = [{"name": "X", "birth_year": by, "calculus_number": input_cn}]
    _fix_calculus_numbers(persons)
    got = persons[0]["calculus_number"]
    if got == expected_cn:
        ok(label)
    else:
        fail(label, f"got {got}, expected {expected_cn}")

# ── _parse_response ───────────────────────────────────────────────────────────
print("--- _parse_response")

# basic clean JSON — two persons with a person-to-person relationship
raw = json.dumps({
    "persons": [
        {"name": "Auguste Laurent", "birth_year": 1807, "death_year": 1853,
         "calculus_number": 4, "nationality": "French", "occupation": "chemist",
         "evidence": "in_source_text_direct"},
        {"name": "Charles Gerhardt", "birth_year": 1816, "death_year": 1856,
         "calculus_number": 4, "nationality": "French", "occupation": "chemist",
         "evidence": "in_source_text_direct"},
    ],
    "relationships": [
        {"subject": "Auguste Laurent", "predicate": "collaborated with",
         "object": "Charles Gerhardt", "evidence": "in_source_text_direct"},
    ]
})
result = _parse_response(raw)
if (result and len(result["persons"]) == 2
        and result["persons"][0]["name"] == "Auguste Laurent"
        and len(result["relationships"]) == 1):
    ok("parse clean JSON")
else:
    fail("parse clean JSON", str(result))

# fenced JSON
fenced = "```json\n" + raw + "\n```"
r2 = _parse_response(fenced)
if r2 and len(r2["persons"]) == 2 and len(r2["relationships"]) == 1:
    ok("parse fenced JSON")
else:
    fail("parse fenced JSON", str(r2))

# dedup + fix_calculus integrated
raw_dedup = json.dumps({
    "persons": [
        {"name": "Edward Mazria", "birth_year": 1942, "death_year": None,
         "calculus_number": 4, "nationality": "American",
         "occupation": "architect", "evidence": "in_source_text_direct"},
        {"name": "Edward Mazria", "birth_year": 1942, "death_year": None,
         "calculus_number": 4, "nationality": "American",
         "occupation": "architect", "evidence": "in_source_text_direct"},
    ],
    "relationships": []
})
r3 = _parse_response(raw_dedup)
if r3 is None:
    fail("parse dedup+fix", "returned None")
elif len(r3["persons"]) != 1:
    fail("parse dedup", f"expected 1 person, got {len(r3['persons'])}")
elif r3["persons"][0]["calculus_number"] != 6:
    fail("parse fix_cn", f"expected cn=6, got {r3['persons'][0]['calculus_number']}")
else:
    ok("parse: dedup + calculus_number fix integrated")

# evidence normalisation
raw_ev = json.dumps({"persons": [{"name": "X", "evidence": "bogus_kind"}], "relationships": []})
r4 = _parse_response(raw_ev)
if r4 and r4["persons"][0]["evidence"] == "llm_training_knowledge":
    ok("parse evidence normalised to llm_training_knowledge")
else:
    fail("parse evidence normalised", str(r4))

# non-JSON input
if _parse_response("I cannot help with that.") is None:
    ok("parse non-JSON returns None")
else:
    fail("parse non-JSON returns None")

# empty string
if _parse_response("") is None:
    ok("parse empty string returns None")
else:
    fail("parse empty string returns None")

# missing keys get defaulted
raw_missing = json.dumps({"persons": [{"name": "X", "evidence": "in_source_text_direct"}]})
r5 = _parse_response(raw_missing)
if r5 and "relationships" in r5 and r5["relationships"] == []:
    ok("parse missing relationships defaulted to []")
else:
    fail("parse missing relationships", str(r5))

# ── _call_one_model guard for empty choices ───────────────────────────────────
print("--- _call_one_model guards")
from graph_builder import _call_one_model
from unittest.mock import patch, MagicMock

# empty choices → return None, no crash
mock_resp = MagicMock()
mock_resp.choices = []
with patch("openai.OpenAI") as mock_openai:
    mock_openai.return_value.chat.completions.create.return_value = mock_resp
    result = _call_one_model("test prompt", "test-model", "fake-key")
if result is None:
    ok("_call_one_model: empty choices returns None")
else:
    fail("_call_one_model: empty choices", str(result))

# None choices → return None, no crash
mock_resp2 = MagicMock()
mock_resp2.choices = None
with patch("openai.OpenAI") as mock_openai:
    mock_openai.return_value.chat.completions.create.return_value = mock_resp2
    result2 = _call_one_model("test prompt", "test-model", "fake-key")
if result2 is None:
    ok("_call_one_model: None choices returns None")
else:
    fail("_call_one_model: None choices", str(result2))

# ── era_distribution ──────────────────────────────────────────────────────────
print("--- era_distribution")

d = era_distribution([
    {"calculus_number": 6}, {"calculus_number": 6},
    {"calculus_number": 4}, {"calculus_number": None},
])
if d == {"Calc6": 2, "Calc4": 1}:
    ok("era_distribution")
else:
    fail("era_distribution", str(d))

if era_distribution([]) == {}:
    ok("era_distribution empty")
else:
    fail("era_distribution empty")

# ── save_graph_state atomic write ─────────────────────────────────────────────
print("--- save_graph_state")

orig_gs = gb._GRAPH_STATE
tmp_dir = pathlib.Path(tempfile.mkdtemp())
gb._GRAPH_STATE = tmp_dir / "graph_state.json"
try:
    state = {"schema_version": "1.0", "files": {"x.pdf": {"status": "complete"}}}
    save_graph_state(state)
    loaded = json.loads((tmp_dir / "graph_state.json").read_text(encoding="utf-8"))
    tmp_gone = not (tmp_dir / "graph_state.tmp").exists()
    if loaded["files"]["x.pdf"]["status"] == "complete" and tmp_gone:
        ok("save_graph_state atomic")
    else:
        fail("save_graph_state atomic", f"loaded={loaded}, tmp_gone={tmp_gone}")

    # unicode preserved
    state2 = {"schema_version": "1.0", "files": {"e.pdf": {"name": "Évariste Galois"}}}
    save_graph_state(state2)
    loaded2 = json.loads((tmp_dir / "graph_state.json").read_text(encoding="utf-8"))
    if loaded2["files"]["e.pdf"]["name"] == "Évariste Galois":
        ok("save_graph_state unicode preserved")
    else:
        fail("save_graph_state unicode", str(loaded2))
finally:
    gb._GRAPH_STATE = orig_gs

# ── get_library_root ──────────────────────────────────────────────────────────
print("--- get_library_root")

root = get_library_root()
if root and root.exists():
    ok(f"get_library_root -> {root.name}")
else:
    fail("get_library_root", f"got {root}")

# ── _get_pending ──────────────────────────────────────────────────────────────
print("--- _get_pending")

orig_ps = gb._PIPELINE_STATE
tmp_ps = tmp_dir / "pipeline_state.json"
tmp_ps.write_text(json.dumps({"files": {
    "a.pdf": {"status": "indexed"},
    "b.pdf": {"status": "indexed"},
    "c.pdf": {"status": "indexed"},
    "d.pdf": {"status": "pending"},       # not indexed
    "e.pdf": {"status": "embedded_no_qdrant"},  # Qdrant lock fallback
}}), encoding="utf-8")
gb._PIPELINE_STATE = tmp_ps
try:
    args = argparse.Namespace(file="", queue=False)
    gs = {"files": {
        "a.pdf": {"status": "complete"},  # skip
        "b.pdf": {"status": "error"},     # retry
    }}
    pending = _get_pending(gs, args)
    if set(pending) == {"b.pdf", "c.pdf", "e.pdf"}:
        ok("_get_pending: complete=skip, error=retry, new=include, unindexed=skip, embedded_no_qdrant=include")
    else:
        fail("_get_pending", f"got {set(pending)}, expected " + "{'b.pdf', 'c.pdf', 'e.pdf'}")

    # empty graph_state — all indexed + embedded_no_qdrant returned
    pending2 = _get_pending({"files": {}}, args)
    if set(pending2) == {"a.pdf", "b.pdf", "c.pdf", "e.pdf"}:
        ok("_get_pending empty graph_state returns all indexed+embedded_no_qdrant")
    else:
        fail("_get_pending empty graph_state", str(set(pending2)))

    # retry_count >= MAX_GRAPH_RETRIES → treated as done (abandoned)
    gs_retried = {"files": {
        "b.pdf": {"status": "error", "retry_count": gb.MAX_GRAPH_RETRIES},
        "c.pdf": {"status": "error", "retry_count": gb.MAX_GRAPH_RETRIES - 1},
    }}
    pending_retried = _get_pending(gs_retried, args)
    if "b.pdf" not in pending_retried and "c.pdf" in pending_retried:
        ok("_get_pending: exhausted retry_count excluded, under-limit still included")
    else:
        fail("_get_pending retry cap", f"got {pending_retried}")

    # --file overrides
    args2 = argparse.Namespace(file="specific.pdf", queue=False)
    if _get_pending(gs, args2) == ["specific.pdf"]:
        ok("_get_pending --file override")
    else:
        fail("_get_pending --file override", str(_get_pending(gs, args2)))
finally:
    gb._PIPELINE_STATE = orig_ps

# ── _filter_person_person_rels ────────────────────────────────────────────────
print("--- _filter_person_person_rels")

persons = [{"name": "Alice"}, {"name": "Bob"}]

# person→person kept
d1 = {"persons": persons, "relationships": [{"subject": "Alice", "predicate": "collaborated with", "object": "Bob"}]}
_filter_person_person_rels(d1)
if len(d1["relationships"]) == 1:
    ok("filter: person-person edge kept")
else:
    fail("filter: person-person edge kept", str(d1["relationships"]))

# person→concept dropped
d2 = {"persons": persons, "relationships": [{"subject": "Alice", "predicate": "authored", "object": "Some Book"}]}
_filter_person_person_rels(d2)
if d2["relationships"] == []:
    ok("filter: person-concept edge dropped")
else:
    fail("filter: person-concept edge dropped", str(d2["relationships"]))

# mixed: only person-person survives
d3 = {"persons": persons, "relationships": [
    {"subject": "Alice", "predicate": "collaborated with", "object": "Bob"},
    {"subject": "Alice", "predicate": "authored", "object": "Calculus"},
]}
_filter_person_person_rels(d3)
if len(d3["relationships"]) == 1 and d3["relationships"][0]["object"] == "Bob":
    ok("filter: mixed — only person-person survives")
else:
    fail("filter: mixed", str(d3["relationships"]))

# empty persons → all rels dropped
d4 = {"persons": [], "relationships": [{"subject": "X", "predicate": "knows", "object": "Y"}]}
_filter_person_person_rels(d4)
if d4["relationships"] == []:
    ok("filter: empty persons → all rels dropped")
else:
    fail("filter: empty persons", str(d4["relationships"]))

# case-insensitive match
d5 = {"persons": [{"name": "Évariste Galois"}], "relationships": [
    {"subject": "Chevalier", "predicate": "received letter from", "object": "évariste galois"},
]}
_filter_person_person_rels(d5)
if len(d5["relationships"]) == 1:
    ok("filter: case-insensitive match")
else:
    fail("filter: case-insensitive match", str(d5["relationships"]))

# _parse_response: person→concept rels stripped end-to-end
raw_mixed = json.dumps({
    "persons": [{"name": "Cauchy", "birth_year": 1789, "death_year": 1857, "calculus_number": 3,
                 "nationality": "French", "occupation": "mathematician", "evidence": "in_source_text_direct"},
                {"name": "Gauss",  "birth_year": 1777, "death_year": 1855, "calculus_number": 3,
                 "nationality": "German",  "occupation": "mathematician", "evidence": "in_source_text_direct"}],
    "relationships": [
        {"subject": "Cauchy", "predicate": "corresponded with", "object": "Gauss",
         "evidence": "in_source_text_direct"},
        {"subject": "Cauchy", "predicate": "developed", "object": "complex analysis",
         "evidence": "in_source_text_direct"},
    ]
})
r = _parse_response(raw_mixed)
if r and len(r["relationships"]) == 1 and r["relationships"][0]["object"] == "Gauss":
    ok("_parse_response: person→concept stripped, person→person kept")
else:
    fail("_parse_response: mixed filter", str(r))

# ── summary ───────────────────────────────────────────────────────────────────
print()
print(f"Results: {PASS} passed, {FAIL} failed")
sys.exit(FAIL)
