#!/usr/bin/env python3
"""
index_timeline.py — Index sacred_timeline JSON into a Qdrant collection.

Each PersonRecord becomes a searchable document. Fields are concatenated into
a text chunk that the RAG pipeline can retrieve alongside the PDF library.

Usage:
    python index_timeline.py                           # index sacred_timeline_current.json
    python index_timeline.py --file sacred_timeline_enriched.json
    python index_timeline.py --device cuda             # GPU embeddings
    python index_timeline.py --reset                   # drop + rebuild collection
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message="Failed to obtain server version")
warnings.filterwarnings("ignore", message=".*symlinks.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="fastembed")

# NVIDIA CUDA 12 DLLs for onnxruntime-gpu on Windows
if sys.platform == "win32":
    import site as _site
    for _sp in _site.getsitepackages():
        for _sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cuda_nvrtc/bin"):
            _dll_dir = Path(_sp) / _sub
            if _dll_dir.is_dir():
                try:
                    os.add_dll_directory(str(_dll_dir))
                except Exception:
                    pass

try:
    import onnxruntime as _ort
    _ort.set_default_logger_severity(4)
except Exception:
    pass

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

console = Console()

BASE_DIR = Path(__file__).parent

# ── Config ────────────────────────────────────────────────────────────────────

COLLECTION_NAME = "sacred_timeline"
DENSE_MODEL     = "jinaai/jina-embeddings-v3"
SPARSE_MODEL    = "Qdrant/bm42-all-minilm-l6-v2-attentions"
DENSE_DIM       = 1024
EMBED_BATCH_SIZE = 64

TIMELINE_CANDIDATES = [
    BASE_DIR / "sacred_timeline_enriched.json",
    BASE_DIR.parent / "sacred-timeline" / "public" / "sacred_timeline_current.json",
    BASE_DIR / "sacred_timeline_5-10-2026.json",
]

# ── Model loading ─────────────────────────────────────────────────────────────

_dense_model  = None
_sparse_model = None

def get_dense_model(device: str = "cpu"):
    global _dense_model
    if _dense_model is None:
        from fastembed import TextEmbedding
        _dense_model = TextEmbedding(
            model_name=DENSE_MODEL,
            providers=["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"],
        )
    return _dense_model

def get_sparse_model():
    global _sparse_model
    if _sparse_model is None:
        from fastembed import SparseTextEmbedding
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _sparse_model

# ── Qdrant ────────────────────────────────────────────────────────────────────

def get_qdrant():
    from qdrant_client import QdrantClient
    qdrant_path = BASE_DIR / "qdrant_data"
    qdrant_path.mkdir(exist_ok=True)
    return QdrantClient(path=str(qdrant_path))

def reset_collection(client) -> None:
    from qdrant_client.models import (
        VectorParams, Distance, SparseVectorParams,
        SparseIndexParams, HnswConfigDiff,
    )
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(full_scan_threshold=5000))},
        hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
    )
    console.print(f"[dim]Created collection '{COLLECTION_NAME}'[/dim]")

def collection_exists(client) -> bool:
    try:
        client.get_collection(COLLECTION_NAME)
        return True
    except Exception:
        return False

# ── Text serialisation of a PersonRecord ─────────────────────────────────────

def person_to_text(p: dict) -> str:
    """Convert a PersonRecord dict into a searchable text chunk."""
    lines: list[str] = []

    name = p.get("name", "Unknown")
    birth = p.get("birth_year")
    death = p.get("death_year")
    dates = f"({birth}–{death})" if birth and death else f"({birth})" if birth else ""
    lines.append(f"{name} {dates}".strip())

    calc = p.get("calculus_number")
    if calc is not None:
        lines.append(f"Era: Calculus {calc}")

    for field in ("nationality", "occupation", "field", "institution"):
        val = p.get(field)
        if val:
            lines.append(f"{field.title()}: {val}")

    bio = p.get("biography") or p.get("description") or p.get("summary")
    if bio:
        lines.append(bio[:600])

    for tag_field in ("suggested_math_tags", "suggested_cognitive_tags",
                      "suggested_domain_tags", "suggested_schools"):
        tags = p.get(tag_field)
        if tags:
            label = tag_field.replace("suggested_", "").replace("_", " ").title()
            lines.append(f"{label}: {', '.join(t if isinstance(t, str) else t.get('tag', str(t)) for t in tags[:10])}")

    rels = p.get("semantic_relationships") or []
    if rels:
        rel_strs = []
        for r in rels[:8]:
            target = r.get("target") or r.get("target_name", "")
            rtype  = r.get("relation_type", "related to")
            if target:
                rel_strs.append(f"{rtype} {target}")
        if rel_strs:
            lines.append("Relationships: " + "; ".join(rel_strs))

    flags = p.get("pipeline_flags") or []
    if flags:
        lines.append("Flags: " + ", ".join(flags[:6]))

    return "\n".join(l for l in lines if l.strip())

def person_payload(p: dict) -> dict:
    """Minimal Qdrant payload for a PersonRecord."""
    return {
        "source_type":  "sacred_timeline",
        "person_id":    p.get("person_id") or p.get("id") or "",
        "name":         p.get("name", ""),
        "birth_year":   p.get("birth_year"),
        "death_year":   p.get("death_year"),
        "calculus_number": p.get("calculus_number"),
        "nationality":  p.get("nationality", ""),
        "field":        p.get("field", ""),
        "text":         person_to_text(p),
    }

# ── Indexing ──────────────────────────────────────────────────────────────────

def index_timeline(timeline_path: Path, device: str = "cpu", reset: bool = False) -> int:
    data = json.loads(timeline_path.read_text(encoding="utf-8-sig"))

    # Handle both flat array and enriched {people: [...]} format
    if isinstance(data, list):
        people = data
    elif isinstance(data, dict) and "people" in data:
        people = data["people"]
    else:
        console.print("[red]Unrecognised timeline format — expected array or {people:[...]}[/red]")
        return 0

    console.print(f"[dim]Loaded {len(people):,} persons from {timeline_path.name}[/dim]")

    client = get_qdrant()

    if reset or not collection_exists(client):
        reset_collection(client)
    else:
        existing = client.get_collection(COLLECTION_NAME).points_count or 0
        if existing > 0:
            console.print(
                f"[yellow]Collection '{COLLECTION_NAME}' already has {existing:,} points.[/yellow]\n"
                f"[dim]Use --reset to drop and rebuild.[/dim]"
            )
            return existing

    console.print(f"[dim]Pre-loading {device.upper()} embedding models…[/dim]", end=" ")
    dense  = get_dense_model(device)
    sparse = get_sparse_model()
    console.print("[green]ready[/green]")

    texts    = [person_to_text(p) for p in people]
    payloads = [person_payload(p)  for p in people]

    from qdrant_client.models import PointStruct, SparseVector

    points: list[PointStruct] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        dense_task  = prog.add_task("Dense embeddings",  total=len(texts))
        sparse_task = prog.add_task("Sparse embeddings", total=len(texts))

        # Dense in batches
        dense_vecs: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i:i + EMBED_BATCH_SIZE]
            for v in dense.embed(batch):
                dense_vecs.append(list(v))
            prog.advance(dense_task, len(batch))

        # Sparse in batches
        sparse_vecs: list[dict] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i:i + EMBED_BATCH_SIZE]
            for sv in sparse.embed(batch):
                sparse_vecs.append({"indices": sv.indices.tolist(), "values": sv.values.tolist()})
            prog.advance(sparse_task, len(batch))

    for idx, (dv, sv, payload) in enumerate(zip(dense_vecs, sparse_vecs, payloads)):
        points.append(PointStruct(
            id=idx,
            vector={"dense": dv, "sparse": SparseVector(indices=sv["indices"], values=sv["values"])},
            payload=payload,
        ))

    # Upsert in batches
    UPSERT_BATCH = 256
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TextColumn("{task.completed}/{task.total}"),
                  TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("Upserting to Qdrant", total=len(points))
        for i in range(0, len(points), UPSERT_BATCH):
            batch = points[i:i + UPSERT_BATCH]
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            prog.advance(task, len(batch))

    final_count = client.get_collection(COLLECTION_NAME).points_count or len(points)
    console.print(f"[bold green]✓ Indexed {final_count:,} persons into '{COLLECTION_NAME}'[/bold green]")
    return final_count

# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Index sacred_timeline JSON into Qdrant")
    parser.add_argument("--file",   default=None, help="Path to timeline JSON (auto-detected if omitted)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Embedding device")
    parser.add_argument("--reset",  action="store_true", help="Drop and rebuild the collection")
    args = parser.parse_args()

    if args.file:
        tl_path = Path(args.file)
        if not tl_path.exists():
            console.print(f"[red]File not found: {tl_path}[/red]")
            sys.exit(1)
    else:
        tl_path = next((p for p in TIMELINE_CANDIDATES if p.exists()), None)
        if tl_path is None:
            console.print("[red]No timeline JSON found. Checked:[/red]")
            for p in TIMELINE_CANDIDATES:
                console.print(f"  [dim]{p}[/dim]")
            sys.exit(1)
        console.print(f"[dim]Using {tl_path}[/dim]")

    index_timeline(tl_path, device=args.device, reset=args.reset)

if __name__ == "__main__":
    main()
