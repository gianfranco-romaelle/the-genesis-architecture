import json, sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open("pipeline_state.json", encoding="utf-8") as f:
    raw = json.load(f)
ps = raw.get("files", raw)
c = Counter(v.get("status", "unknown") for v in ps.values())
print("Pipeline state:", dict(c.most_common()))

recent = sorted(
    [(k, v) for k, v in ps.items() if v.get("status") == "indexed"],
    key=lambda x: x[1].get("updated_at", ""),
    reverse=True
)[:5]
print("Most recently indexed:")
for path, info in recent:
    chunks = info.get("chunks_indexed", "?")
    ts = info.get("updated_at", "")[:19]
    print(f"  {path[-55:]}  chunks={chunks}  @{ts}")

with open("graph_state.json", encoding="utf-8") as f:
    raw_g = json.load(f)
gs = raw_g.get("files", raw_g)
gc = Counter(v.get("status", "unknown") for v in gs.values())
print(f"\nGraph state: {dict(gc.most_common())}")
