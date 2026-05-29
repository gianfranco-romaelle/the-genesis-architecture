#!/usr/bin/env python3
"""
One-time DJVU → PDF batch converter for the Auguste Laurent Society library.
Usage:
    python convert_djvu_to_pdf.py [--dry-run] [--delete-originals]
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Ensure Unicode filenames print safely on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

LIBRARY_ROOT = Path(r"G:\Other computers\My Laptop\THE AUGUSTE LAURENT SOCIETY")

EXCLUDED_PATHS = [
    r"Blogs Translations and Friends Libraries\CoreyDigs",
]

DDJVU_CANDIDATES = [
    r"C:\Program Files\DjVuLibre\ddjvu.exe",
    r"C:\Program Files (x86)\DjVuLibre\ddjvu.exe",
    "ddjvu",
]


def find_ddjvu() -> str | None:
    for candidate in DDJVU_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            return str(path)
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def is_excluded(path: Path) -> bool:
    path_str = str(path)
    return any(excl.lower() in path_str.lower() for excl in EXCLUDED_PATHS)


def collect_djvu_files(dry_run: bool) -> list[Path]:
    if not LIBRARY_ROOT.exists():
        print(f"ERROR: Library root not found: {LIBRARY_ROOT}")
        sys.exit(1)

    files = []
    for djvu in LIBRARY_ROOT.rglob("*.djvu"):
        if is_excluded(djvu):
            continue
        pdf_out = djvu.with_suffix(".pdf")
        if pdf_out.exists():
            continue
        files.append(djvu)

    files.sort()
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DJVU files to PDF using DjVuLibre.")
    parser.add_argument("--dry-run", action="store_true", help="List files without converting")
    parser.add_argument("--delete-originals", action="store_true",
                        help="Delete DJVU file after successful conversion")
    args = parser.parse_args()

    files = collect_djvu_files(args.dry_run)
    print(f"Found {len(files)} DJVU files to convert (PDFs already present are skipped)")

    if not files:
        print("Nothing to do.")
        return

    if args.dry_run:
        for f in files:
            try:
                rel = f.relative_to(LIBRARY_ROOT)
            except ValueError:
                rel = f
            print(f"  DRY RUN: {rel}")
        return

    ddjvu = find_ddjvu()
    if not ddjvu:
        print("ERROR: ddjvu not found. Install DjVuLibre from https://djvu.sourceforge.net/")
        sys.exit(1)
    print(f"Using ddjvu: {ddjvu}\n")

    ok = fail = 0
    for i, djvu in enumerate(files, 1):
        try:
            rel = djvu.relative_to(LIBRARY_ROOT)
        except ValueError:
            rel = djvu
        pdf_out = djvu.with_suffix(".pdf")

        print(f"[{i}/{len(files)}] {rel} ...", end=" ", flush=True)
        try:
            result = subprocess.run(
                [ddjvu, "-format=pdf", str(djvu), str(pdf_out)],
                capture_output=True,
                timeout=300,
            )
            if result.returncode == 0:
                print("OK")
                ok += 1
                if args.delete_originals:
                    djvu.unlink()
            else:
                stderr = result.stderr.decode("utf-8", errors="replace").strip()
                print(f"FAIL (exit {result.returncode}: {stderr[:80]})")
                fail += 1
                if pdf_out.exists():
                    pdf_out.unlink()  # remove partial output
        except subprocess.TimeoutExpired:
            print("FAIL (timeout)")
            fail += 1
            if pdf_out.exists():
                pdf_out.unlink()
        except Exception as exc:
            print(f"FAIL ({exc})")
            fail += 1

    print(f"\nDone: {ok} converted, {fail} failed")


if __name__ == "__main__":
    main()
