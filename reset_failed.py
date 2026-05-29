#!/usr/bin/env python3
"""
reset_failed.py — Reset 'failed' pipeline_state entries so build_index retries them.

Run this after Google Drive has finished syncing files offline.
Then run:  python build_index.py --device cuda
"""
import json, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ps_path = Path(__file__).parent / "pipeline_state.json"
ps = json.loads(ps_path.read_text(encoding="utf-8"))

before = sum(1 for v in ps["files"].values() if v.get("status") == "failed")
for v in ps["files"].values():
    if v.get("status") == "failed":
        v.clear()   # remove status, error, updated_at — file will be re-evaluated

tmp = ps_path.with_suffix(".tmp")
tmp.write_text(json.dumps(ps, ensure_ascii=False, indent=2), encoding="utf-8")
tmp.replace(ps_path)

print(f"Reset {before} failed entries — ready to re-index.")
print("Next step: python build_index.py --device cuda")
