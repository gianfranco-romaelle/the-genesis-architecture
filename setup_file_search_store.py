#!/usr/bin/env python3
"""
One-time library indexing into a Gemini File Search Store.
Run after convert_djvu_to_pdf.py.

Usage:
    python setup_file_search_store.py [--dry-run] [--resume]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
PIPELINE_STATE_PATH = BASE_DIR / "pipeline_state.json"

LIBRARY_ROOT = Path(r"G:\Other computers\My Laptop\THE AUGUSTE LAURENT SOCIETY")

EXCLUDED_PATHS = [
    r"Blogs Translations and Friends Libraries\CoreyDigs",
]

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".doc", ".html", ".htm"}

STORE_DISPLAY_NAME = "genesis-architecture-auguste-laurent"

COST_PER_1M_TOKENS = 0.15
APPROX_TOKENS_PER_FILE = 100_000


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------
def load_state() -> dict:
    if PIPELINE_STATE_PATH.exists():
        with open(PIPELINE_STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict) -> None:
    with open(PIPELINE_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------
def is_excluded(path: Path) -> bool:
    path_str = str(path)
    return any(excl.lower() in path_str.lower() for excl in EXCLUDED_PATHS)


def collect_files() -> list[Path]:
    if not LIBRARY_ROOT.exists():
        print(f"ERROR: Library root not found: {LIBRARY_ROOT}")
        sys.exit(1)

    files = []
    for path in LIBRARY_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if is_excluded(path):
            continue
        files.append(path)

    files.sort()
    return files


def fmt_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes //= 1024
    return f"{n_bytes:.1f} TB"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Index the Auguste Laurent Society library into a Gemini File Search Store.")
    parser.add_argument("--dry-run", action="store_true", help="List files + estimate cost without uploading")
    parser.add_argument("--resume", action="store_true", help="Resume from existing store in pipeline_state.json")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        env_file = BASE_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key and not args.dry_run:
        print("ERROR: GEMINI_API_KEY not set")
        sys.exit(1)

    files = collect_files()
    total_bytes = sum(f.stat().st_size for f in files)
    estimated_cost = len(files) * APPROX_TOKENS_PER_FILE / 1_000_000 * COST_PER_1M_TOKENS

    print(f"Files found:     {len(files)}")
    print(f"Total size:      {fmt_size(total_bytes)}")
    print(f"Estimated cost:  ${estimated_cost:.2f}")
    print()

    if args.dry_run:
        print("DRY RUN — first 20 files:")
        for f in files[:20]:
            try:
                rel = f.relative_to(LIBRARY_ROOT)
            except ValueError:
                rel = f
            print(f"  {rel}  ({fmt_size(f.stat().st_size)})")
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more")
        print(f"\nEstimated indexing cost: ${estimated_cost:.2f}")
        return

    # Import SDK
    try:
        from google import genai
    except ImportError:
        print("ERROR: google-genai not installed. Run: pip install google-genai")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    state = load_state()

    # Ensure state structure
    state.setdefault("file_search_store", {
        "store_name": None,
        "indexed_files": [],
        "failed_files": [],
    })

    fss = state["file_search_store"]
    store_name: str | None = fss.get("store_name")
    indexed_set = set(fss.get("indexed_files", []))

    # Create or reuse store
    if args.resume and store_name:
        print(f"Resuming with existing store: {store_name}")
    else:
        print(f"Creating File Search Store '{STORE_DISPLAY_NAME}' ...")
        store = client.file_search_stores.create(
            config={"display_name": STORE_DISPLAY_NAME}
        )
        store_name = store.name
        fss["store_name"] = store_name
        save_state(state)
        print(f"  Created: {store_name}\n")

    # Filter out already-done files
    pending = [f for f in files if str(f) not in indexed_set]
    print(f"Uploading {len(pending)} files ({len(indexed_set)} already indexed) ...")
    print()

    ok = fail = 0
    for i, filepath in enumerate(pending, 1):
        try:
            rel = filepath.relative_to(LIBRARY_ROOT)
        except ValueError:
            rel = filepath
        size_str = fmt_size(filepath.stat().st_size)

        print(f"[{i}/{len(pending)}] {rel} ({size_str}) ...", end=" ", flush=True)
        try:
            operation = client.file_search_stores.upload_to_file_search_store(
                file=str(filepath),
                file_search_store_name=store_name,
                config={
                    "display_name": filepath.name,
                    "metadata": {
                        "subfolder": str(rel.parent),
                        "extension": filepath.suffix.lower(),
                    },
                },
            )
            # Poll for completion
            retries = 0
            while not operation.done:
                time.sleep(3)
                try:
                    operation = client.operations.get(operation)
                except Exception:
                    retries += 1
                    if retries > 20:
                        raise RuntimeError("Operation polling timed out")
                    time.sleep(5)

            print("OK")
            ok += 1
            fss["indexed_files"].append(str(filepath))
            save_state(state)

        except Exception as exc:
            print(f"FAIL ({exc})")
            fail += 1
            fss["failed_files"].append(str(filepath))
            save_state(state)

    print()
    print("=" * 60)
    print(f"Indexing complete: {ok} uploaded, {fail} failed")
    print()
    print(f"STORE NAME: {store_name}")
    print()
    print("This store name has been saved to pipeline_state.json.")
    print("The pipeline will use it automatically.")
    save_state(state)


if __name__ == "__main__":
    main()
