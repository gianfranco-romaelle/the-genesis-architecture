#!/usr/bin/env python
"""Quick smoke test: builder + latex on one small PDF."""
from pathlib import Path
import sys

candidates = [p for p in Path("G:/").rglob("*.pdf") if p.stat().st_size < 200_000][:1] if Path("G:/").exists() else []
if not candidates:
    print("no small pdf found on G: — skipping live PDF test"); sys.exit(0)

src = candidates[0]
print(f"testing with: {src.name} ({src.stat().st_size // 1024} KB)")

import histodoc_builder as hb
doc = hb.process_file(src, source_lang="eng")
print(f"  regions : {len(doc['regions'])}")
print(f"  layers  : {len(doc['layers'])}")
print(f"  template: {doc['latex_template']}")
types: dict = {}
for r in doc["regions"]:
    types[r["region_type"]] = types.get(r["region_type"], 0) + 1
print(f"  region types: {types}")

# verify three layers per region
rids = set(r["canonical_region_id"] for r in doc["regions"])
for rid in list(rids)[:3]:
    lyrs = {l["layer_type"] for l in doc["layers"] if l["canonical_region_id"] == rid}
    assert lyrs == {"diplomatic", "normalized", "translation"}, f"missing layers for {rid}: {lyrs}"
print("  layer coverage: OK")

import histodoc_latex as hl
tex = hl.render_latex(doc)
print(f"  LaTeX: {len(tex):,} chars, {tex.count(chr(10))} lines")
assert r"\begin{document}" in tex, "missing \\begin{document}"
assert r"\end{document}" in tex,   "missing \\end{document}"
assert r"\section" in tex or len(doc["regions"]) == 0, "no \\section in output"
print("  LaTeX structure: OK")

# test _differs helper
from histodoc_latex import _differs
assert _differs("word-\nedge", "wordedge") is True
assert _differs("hello world", "hello world") is False
assert _differs("hello  world", "hello world") is False   # whitespace only
print("  _differs: OK")

# test _tex escaping
from histodoc_latex import _tex
assert _tex("50% & 100$") == r"50\% \& 100\$"
print("  _tex escape: OK")

print("\nPASS — all assertions green")
