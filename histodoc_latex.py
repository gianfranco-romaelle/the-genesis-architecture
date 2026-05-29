#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
histodoc_latex.py — generate LaTeX from a HistoDoc JSON document.

Rendering model
  • normalized layer → main body text
  • diplomatic layer → footnote when it materially differs from normalized
    (this is standard critical-edition practice)
  • translation layer → used only when source_language != 'eng';
    rendered as a parallel column via the reledpar package if template='reledpar'
  • heading / subheading → \\section / \\subsection
  • footnote regions   → \\footnote inline with the preceding paragraph
  • margin_note        → \\marginpar
  • running_title      → suppressed (already in the original header/footer)

Templates
  article   standard academic article   \\documentclass{article}
  memoir    long-form book              \\documentclass{memoir}
  reledpar  critical parallel-text edition  \\documentclass{reledpar}

Usage
  python histodoc_latex.py doc.histodoc.json
  python histodoc_latex.py doc.histodoc.json --output out.tex
  python histodoc_latex.py doc.histodoc.json --template memoir
  python histodoc_latex.py --all               # all parsed files in histodoc_state.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

_HERE       = Path(__file__).parent
_STATE_PATH = _HERE / "histodoc_state.json"
_OUT_DIR    = _HERE / "histodoc"

# ── LaTeX escape ──────────────────────────────────────────────────────────────

_TEX_ESCAPE = str.maketrans({
    "&":  r"\&",
    "%":  r"\%",
    "$":  r"\$",
    "#":  r"\#",
    "_":  r"\_",
    "{":  r"\{",
    "}":  r"\}",
    "~":  r"\textasciitilde{}",
    "^":  r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
})


def _tex(s: str) -> str:
    """Escape a plain-text string for LaTeX body use."""
    if not s:
        return ""
    return s.translate(_TEX_ESCAPE)


# ── differ: does diplomatic differ meaningfully from normalized? ──────────────

_WS_RE = re.compile(r"\s+")


def _differs(dipl: str, norm: str) -> bool:
    """True if diplomatic differs from normalized beyond whitespace."""
    d = _WS_RE.sub(" ", dipl).strip()
    n = _WS_RE.sub(" ", norm).strip()
    return d != n


# ── document index helpers ────────────────────────────────────────────────────

def _build_index(doc: dict) -> tuple[dict, dict]:
    """
    Returns:
      regions_by_id : canonical_region_id → region dict
      layers_by_id  : canonical_region_id → {layer_type: text, ...}
    """
    regions_by_id: dict = {r["canonical_region_id"]: r for r in doc.get("regions", [])}
    layers_by_id:  dict = {}
    for lyr in doc.get("layers", []):
        rid = lyr["canonical_region_id"]
        layers_by_id.setdefault(rid, {})[lyr["layer_type"]] = lyr.get("text", "")
    return regions_by_id, layers_by_id


# ── preamble builders ─────────────────────────────────────────────────────────

def _preamble_article(doc: dict) -> str:
    title  = _tex(doc.get("title")  or "Untitled")
    author = doc.get("author") or ""
    if isinstance(author, list):
        author = " and ".join(author)
    author = _tex(author)
    year   = doc.get("year") or ""

    return (
        r"\documentclass[12pt]{article}" "\n"
        r"\usepackage[T1]{fontenc}" "\n"
        r"\usepackage[utf8]{inputenc}" "\n"
        r"\usepackage{lmodern}" "\n"
        r"\usepackage{microtype}" "\n"
        r"\usepackage{csquotes}" "\n"
        r"\usepackage[margin=1in]{geometry}" "\n"
        r"\usepackage{setspace}" "\n"
        r"\onehalfspacing" "\n"
        r"\usepackage{marginnote}" "\n"
        "\n"
        rf"\title{{{title}}}" "\n"
        rf"\author{{{author}}}" "\n"
        rf"\date{{{year}}}" "\n"
        r"\begin{document}" "\n"
        r"\maketitle" "\n"
    )


def _preamble_memoir(doc: dict) -> str:
    title  = _tex(doc.get("title")  or "Untitled")
    author = doc.get("author") or ""
    if isinstance(author, list):
        author = " and ".join(author)
    author = _tex(author)
    year   = doc.get("year") or ""

    return (
        r"\documentclass[12pt,oneside]{memoir}" "\n"
        r"\usepackage[T1]{fontenc}" "\n"
        r"\usepackage[utf8]{inputenc}" "\n"
        r"\usepackage{lmodern}" "\n"
        r"\usepackage{microtype}" "\n"
        r"\usepackage{csquotes}" "\n"
        r"\usepackage{marginnote}" "\n"
        "\n"
        rf"\title{{{title}}}" "\n"
        rf"\author{{{author}}}" "\n"
        rf"\date{{{year}}}" "\n"
        r"\begin{document}" "\n"
        r"\frontmatter" "\n"
        r"\maketitle" "\n"
        r"\mainmatter" "\n"
    )


def _preamble_reledpar(doc: dict) -> str:
    title  = _tex(doc.get("title")  or "Untitled")
    author = doc.get("author") or ""
    if isinstance(author, list):
        author = " and ".join(author)
    author = _tex(author)
    year   = doc.get("year") or ""

    return (
        r"\documentclass[12pt]{article}" "\n"
        r"\usepackage[T1]{fontenc}" "\n"
        r"\usepackage[utf8]{inputenc}" "\n"
        r"\usepackage{lmodern}" "\n"
        r"\usepackage{reledpar}" "\n"
        r"\usepackage{reledmac}" "\n"
        r"\usepackage{csquotes}" "\n"
        r"\usepackage[margin=1in]{geometry}" "\n"
        "\n"
        rf"\title{{{title}}}" "\n"
        rf"\author{{{author}}}" "\n"
        rf"\date{{{year}}}" "\n"
        r"\begin{document}" "\n"
        r"\maketitle" "\n"
    )


_PREAMBLES = {
    "article":  _preamble_article,
    "memoir":   _preamble_memoir,
    "reledpar": _preamble_reledpar,
}


# ── body renderer ─────────────────────────────────────────────────────────────

def _render_body(doc: dict, template: str) -> str:
    regions_by_id, layers_by_id = _build_index(doc)
    src_lang = doc.get("source_language", "eng")

    # sort regions by sequence_index
    regions_sorted = sorted(
        doc.get("regions", []),
        key=lambda r: r.get("sequence_index", 0),
    )

    lines: list[str] = []
    pending_footnotes: list[str] = []  # author footnotes collected per paragraph

    for region in regions_sorted:
        rid  = region["canonical_region_id"]
        rtyp = region.get("region_type", "paragraph")
        lyrs = layers_by_id.get(rid, {})

        norm = lyrs.get("normalized", "").strip()
        dipl = lyrs.get("diplomatic", "").strip()
        tran = lyrs.get("translation", "").strip()

        if not norm and not dipl:
            continue

        # choose display text: normalized for body; diplomatic as footnote if differs
        body_text = norm or dipl
        dipl_note = ""
        if dipl and norm and _differs(dipl, norm):
            dipl_note = rf"\footnote{{Diplomatic: {_tex(dipl)}}}"

        if rtyp == "running_title":
            continue  # suppress page headers/footers

        elif rtyp == "heading":
            lines.append(rf"\section{{{_tex(body_text)}}}")

        elif rtyp == "subheading":
            lines.append(rf"\subsection{{{_tex(body_text)}}}")

        elif rtyp == "footnote":
            # Author's original footnote: collect and emit after current paragraph
            pending_footnotes.append(_tex(body_text))

        elif rtyp == "margin_note":
            lines.append(rf"\marginnote{{{_tex(body_text)}}}")

        elif rtyp == "caption":
            lines.append(rf"\caption{{{_tex(body_text)}}}")

        elif rtyp in ("paragraph", "list_item", "epigraph",
                      "mathematical_expression", "colophon", "table_cell"):
            # For reledpar with non-English source: parallel columns
            if template == "reledpar" and src_lang != "eng" and tran:
                lines.append(r"\begin{pairs}")
                lines.append(r"\begin{Leftside}")
                lines.append(rf"\pstart {_tex(body_text)}{dipl_note} \pend")
                lines.append(r"\end{Leftside}")
                lines.append(r"\begin{Rightside}")
                lines.append(rf"\pstart {_tex(tran)} \pend")
                lines.append(r"\end{Rightside}")
                lines.append(r"\Columns")
                lines.append(r"\end{pairs}")
            else:
                # flush pending footnotes inline into this paragraph
                fn_block = "".join(
                    rf"\footnote{{[Author's note] {fn}}}"
                    for fn in pending_footnotes
                )
                pending_footnotes = []
                lines.append(
                    rf"{_tex(body_text)}{dipl_note}{fn_block}" "\n"
                )

        else:
            # fallback: emit as paragraph
            lines.append(rf"{_tex(body_text)}" "\n")

    # flush any trailing footnotes
    if pending_footnotes:
        for fn in pending_footnotes:
            lines.append(rf"\footnote{{[Author's note] {_tex(fn)}}}")

    return "\n".join(lines)


# ── main assembler ────────────────────────────────────────────────────────────

def render_latex(doc: dict, template: str | None = None) -> str:
    tmpl = template or doc.get("latex_template", "article")
    if tmpl not in _PREAMBLES:
        tmpl = "article"

    preamble = _PREAMBLES[tmpl](doc)
    body     = _render_body(doc, tmpl)
    comment  = (
        "% Generated by histodoc_latex.py — normalized layer as body text\n"
        "% Diplomatic variants appear as \\footnote where text differs.\n"
        f"% Source: {doc.get('source_path', '')}\n"
        f"% Generated: {datetime.now(timezone.utc).isoformat()}\n"
    )

    return comment + preamble + "\n" + body + "\n\\end{document}\n"


# ── state helpers ─────────────────────────────────────────────────────────────

def _load_state() -> dict:
    try:
        return json.loads(_STATE_PATH.read_bytes().decode("utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {"files": {}}


def _mark_latex(state: dict, src_path: str, tex_path: str) -> None:
    entry = state.get("files", {}).get(src_path, {})
    entry["latex_path"]  = tex_path
    entry["status"]      = "latex_built"
    entry["updated_at"]  = datetime.now(timezone.utc).isoformat()
    state.setdefault("files", {})[src_path] = entry
    tmp = _STATE_PATH.with_suffix(".tmp")
    tmp.write_bytes(json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8"))
    os.replace(tmp, _STATE_PATH)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    ap = argparse.ArgumentParser(description="HistoDoc → LaTeX renderer")
    ap.add_argument(
        "input", nargs="?", default=None,
        help="Path to a .histodoc.json file",
    )
    ap.add_argument("--output", "-o", default=None, help="Output .tex path")
    ap.add_argument(
        "--template", choices=["article", "memoir", "reledpar"], default=None,
        help="Override document template",
    )
    ap.add_argument(
        "--all", action="store_true",
        help="Build LaTeX for all 'parsed' entries in histodoc_state.json",
    )
    return ap.parse_args()


def _build_one(json_path: Path, out_path: Path, template: str | None) -> None:
    doc = json.loads(json_path.read_bytes().decode("utf-8", errors="replace"))
    tex = render_latex(doc, template)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tex.encode("utf-8"))
    print(f"  → {out_path}  ({len(tex):,} chars)", flush=True)


def main():
    args  = _parse_args()
    state = _load_state()

    if args.all:
        targets = [
            (Path(e.get("source_path", "")), e)
            for e in state.get("files", {}).values()
            if e.get("status") in ("parsed", "translated")
        ]
        if not targets:
            print("No parsed HistoDoc files in state.", flush=True)
            return
        print(f"Building LaTeX for {len(targets)} documents…", flush=True)
        for src_path, entry in targets:
            doc_id   = entry.get("document_id", "")
            stem     = re.sub(r"[^a-zA-Z0-9_\-]", "_", src_path.stem)[:120]
            json_p   = _OUT_DIR / f"{stem}.histodoc.json"
            tex_p    = _OUT_DIR / f"{stem}.tex"
            if not json_p.exists():
                print(f"  SKIP (json missing): {json_p.name}", flush=True)
                continue
            print(f"  {src_path.name} … ", end="", flush=True)
            try:
                _build_one(json_p, tex_p, args.template)
                _mark_latex(state, str(src_path), str(tex_p))
            except Exception as exc:
                print(f"ERROR: {exc}", flush=True)
        return

    if not args.input:
        print("Provide a .histodoc.json path or use --all", flush=True)
        sys.exit(1)

    json_p = Path(args.input)
    if not json_p.exists():
        print(f"Not found: {json_p}", flush=True)
        sys.exit(1)

    if args.output:
        tex_p = Path(args.output)
    else:
        tex_p = json_p.with_suffix(".tex")

    _build_one(json_p, tex_p, args.template)

    # update state if this is a tracked file
    doc = json.loads(json_p.read_bytes().decode("utf-8", errors="replace"))
    src = doc.get("source_path", "")
    if src in state.get("files", {}):
        _mark_latex(state, src, str(tex_p))


if __name__ == "__main__":
    main()
