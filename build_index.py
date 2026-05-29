#!/usr/bin/env python3
"""
build_index.py — Ingestion pipeline for the Auguste Laurent Society library.

Parses PDFs → detects language → translates to English → chunks into
parent/child pairs → embeds (Jina v3 dense + BM42 sparse) → indexes into
a local Qdrant collection. Fully resumable via SHA-256 dirty tracking.

Usage:
    python build_index.py [--dry-run] [--limit N] [--force-reindex]
                          [--library-root PATH] [--zotero-export PATH]
                          [--qdrant-url URL] [--collection NAME]
                          [--device cpu|cuda|mps]

Requirements (install in this order):
    pip install qdrant-client fastembed langdetect tiktoken rich anthropic pymupdf
    # OCR / layout parsing (optional — adds quality on scanned PDFs):
    pip install docling
    pip install marker-pdf          # may require poppler/pdfium on Windows

Qdrant must be running locally before indexing:
    docker run -p 6333:6333 -v %USERPROFILE%/qdrant_storage:/qdrant/storage qdrant/qdrant
    # or download the Windows binary: https://github.com/qdrant/qdrant/releases
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

# Ensure Unicode prints safely on Windows CP1252 consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Rich (always required) ────────────────────────────────────────────────────
import os
import warnings

# Suppress noisy but harmless warnings from HF Hub and Qdrant client
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# Allow ONNX runtime to use all available cores for embedding throughput
import multiprocessing as _mp
_cpu_count = str(_mp.cpu_count())
os.environ.setdefault("OMP_NUM_THREADS", _cpu_count)
os.environ.setdefault("ONNXRUNTIME_NUM_THREADS", _cpu_count)
warnings.filterwarnings("ignore", message="Failed to obtain server version")
warnings.filterwarnings("ignore", message=".*symlinks.*")
# Suppress fastembed's noisy "Attempt to set CUDAExecutionProvider failed" RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning, module="fastembed")

# Add NVIDIA CUDA 12 DLL directories so onnxruntime-gpu finds cublasLt64_12.dll
import sys as _sys
if _sys.platform == "win32":
    import site as _site
    for _sp in _site.getsitepackages():
        for _sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cuda_nvrtc/bin"):
            _dll_dir = Path(_sp) / _sub
            if _dll_dir.is_dir():
                try:
                    os.add_dll_directory(str(_dll_dir))
                except Exception:
                    pass

from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress,
    TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table

console = Console()

# ── Optional imports with graceful fallback ───────────────────────────────────
try:
    import fitz  # PyMuPDF
    PYMUPDF_OK = True
except ImportError:
    PYMUPDF_OK = False

try:
    from docling.document_converter import DocumentConverter
    DOCLING_OK = True
except ImportError:
    DOCLING_OK = False

try:
    from marker.convert import convert_single_pdf
    from marker.models import load_all_models
    MARKER_OK = True
except ImportError:
    MARKER_OK = False

try:
    from langdetect import detect as _langdetect
    from langdetect.lang_detect_exception import LangDetectException
    LANGDETECT_OK = True
except ImportError:
    LANGDETECT_OK = False

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
    TIKTOKEN_OK = True
except ImportError:
    def count_tokens(text: str) -> int:
        return len(text.split()) * 4 // 3  # rough fallback
    TIKTOKEN_OK = False

try:
    import onnxruntime as _ort
    _ort.set_default_logger_severity(4)  # suppress CUDA init noise (FATAL only)
except Exception:
    pass

try:
    from fastembed import TextEmbedding, SparseTextEmbedding
    FASTEMBED_OK = True
except ImportError:
    FASTEMBED_OK = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm
    QDRANT_OK = True
except ImportError:
    QDRANT_OK = False

try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_OK = True
except ImportError:
    TRANSFORMERS_OK = False

try:
    import requests as _requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

# djvulibre CLI tools — check PATH first, then Windows default install location
def _find_djvulibre_exe(name: str) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    # winget / NSIS installer puts tools here on Windows (x86 even on x64)
    for base in [
        r"C:\Program Files (x86)\DjVuLibre",
        r"C:\Program Files\DjVuLibre",
    ]:
        candidate = Path(base) / f"{name}.exe"
        if candidate.exists():
            return str(candidate)
    return None

_DDJVU = _find_djvulibre_exe("ddjvu")
_DJVUTXT = _find_djvulibre_exe("djvutxt")
DJVU_CLI_OK = bool(_DDJVU or _DJVUTXT)

# ── Configuration ─────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent

LIBRARY_ROOT = Path(r"G:\Other computers\My Laptop\THE AUGUSTE LAURENT SOCIETY")
STATE_FILE = _HERE / "pipeline_state.json"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "als_library"
DENSE_MODEL = "jinaai/jina-embeddings-v3"
SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
DENSE_DIM = 1024

PARENT_CHUNK_TOKENS = 1024
CHILD_CHUNK_TOKENS = 256
CHUNK_OVERLAP_TOKENS = 32
MIN_CHUNK_CHARS = 80       # discard chunks shorter than this
SCANNED_CHAR_DENSITY = 0.3  # chars-per-byte below this → treat as scanned

SUPPORTED_EXTENSIONS = {".pdf", ".djvu"}

JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
JINA_EMBED_MODEL = "jina-embeddings-v3"
JINA_MAX_BATCH = 128
EXCLUDED_PATHS = [
    r"Blogs Translations and Friends Libraries\CoreyDigs",
]

TRANSLATION_MODELS: dict[str, str] = {
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "de": "Helsinki-NLP/opus-mt-de-en",
    "it": "Helsinki-NLP/opus-mt-it-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "pt": "Helsinki-NLP/opus-mt-pt-en",
    "nl": "Helsinki-NLP/opus-mt-nl-en",
    "la": "Helsinki-NLP/opus-mt-itc-en",
}

# ── State management ──────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        with STATE_FILE.open(encoding="utf-8") as f:
            return json.load(f)
    return {"files": {}, "collection": COLLECTION_NAME, "schema_version": "1.0.0"}

def save_state(state: dict) -> None:
    with STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()

def read_file_bytes(path: Path) -> bytes:
    """Read file once into memory for hash + PDF parse in a single network round-trip."""
    with path.open("rb") as f:
        return f.read()

def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def is_excluded(path: Path) -> bool:
    s = str(path)
    return any(excl.lower() in s.lower() for excl in EXCLUDED_PATHS)

# ── Zotero metadata ───────────────────────────────────────────────────────────

def load_zotero_index(export_path: Optional[Path]) -> dict[str, dict]:
    """
    Returns a dict keyed by normalized file basename → Zotero item metadata.
    Falls back to empty dict if no export provided.
    """
    if not export_path or not export_path.exists():
        return {}
    with export_path.open(encoding="utf-8") as f:
        data = json.load(f)
    items = data if isinstance(data, list) else data.get("items", [])
    index: dict[str, dict] = {}
    for item in items:
        for att in item.get("attachments", []):
            p = att.get("path", "")
            if p:
                index[Path(p).stem.lower()] = item
    return index

def zotero_metadata(path: Path, zot: dict[str, dict]) -> dict:
    key = path.stem.lower()
    item = zot.get(key, {})
    return {
        "zotero_key": item.get("citekey", ""),
        "title": item.get("title", path.stem),
        "author": item.get("author", ""),
        "year": item.get("date", "")[:4] if item.get("date") else None,
        "language": item.get("language", ""),
        "publication_type": item.get("type", "other"),
    }

# ── PDF parsing ───────────────────────────────────────────────────────────────

def _is_scanned(text: str, file_size: int) -> bool:
    if file_size == 0:
        return True
    density = len(text) / file_size
    return density < SCANNED_CHAR_DENSITY

def _normalize_page_text(text: str) -> str:
    """
    PyMuPDF inserts \n at every line break, including within sentences.
    Collapse intra-paragraph line breaks to spaces; preserve paragraph breaks
    (double newlines) as sentence boundaries.
    """
    # Preserve double newlines as paragraph separators
    text = re.sub(r'\n{2,}', '\x00PARA\x00', text)
    # Single newline inside a word (hyphenated line-break): rejoin
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    # Remaining single newlines → space
    text = re.sub(r'\n', ' ', text)
    # Restore paragraph breaks as double newlines
    text = text.replace('\x00PARA\x00', '\n\n')
    # Collapse multiple spaces
    text = re.sub(r'  +', ' ', text)
    return text.strip()

def parse_pdf_pymupdf(path: Path, raw: Optional[bytes] = None) -> list[dict]:
    """Fast extraction for digital-native PDFs. Returns list of page dicts.
    Pass raw bytes to avoid a second file read over network drives."""
    if not PYMUPDF_OK:
        return []
    doc = fitz.open(stream=raw, filetype="pdf") if raw else fitz.open(str(path))
    pages = []
    for i, page in enumerate(doc):
        text = _normalize_page_text(page.get_text("text"))
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages

def parse_pdf_docling(path: Path) -> list[dict]:
    """Layout-aware parsing for scanned / complex PDFs via Docling."""
    if not DOCLING_OK:
        return []
    try:
        conv = DocumentConverter()
        result = conv.convert(str(path))
        md = result.document.export_to_markdown()
        # Split by page markers if present, otherwise treat as single page
        sections = re.split(r"\n#{1,3} Page \d+", md)
        return [
            {"page": i + 1, "text": s.strip()}
            for i, s in enumerate(sections) if s.strip()
        ]
    except Exception as exc:
        console.print(f"  [yellow]Docling error on {path.name}: {exc}[/yellow]")
        return []

def parse_pdf(path: Path) -> list[dict]:
    """
    Parse strategy:
      1. PyMuPDF fast extraction
      2. If density too low (scanned) → Docling for OCR
      3. If both fail → return empty (file logged as failed)
    """
    pages = parse_pdf_pymupdf(path)
    full_text = " ".join(p["text"] for p in pages)
    if _is_scanned(full_text, path.stat().st_size) and DOCLING_OK:
        docling_pages = parse_pdf_docling(path)
        if docling_pages:
            return docling_pages
    return pages

def parse_pdf_from_bytes(raw: bytes, path: Path) -> list[dict]:
    """Parse PDF from already-loaded bytes (avoids re-reading from network)."""
    pages = parse_pdf_pymupdf(path, raw=raw)
    full_text = " ".join(p["text"] for p in pages)
    if _is_scanned(full_text, len(raw)) and DOCLING_OK:
        docling_pages = parse_pdf_docling(path)
        if docling_pages:
            return docling_pages
    return pages

def parse_djvu(path: Path) -> list[dict]:
    """
    Extract text from a DJVU file using djvulibre CLI tools.
    Tries ddjvu (→ PDF → PyMuPDF) first for best layout; falls back to djvutxt.
    Install: conda install -c conda-forge djvulibre
    """
    if _DDJVU:
        tmp = Path(tempfile.mktemp(suffix=".pdf"))
        try:
            r = subprocess.run(
                [_DDJVU, "-format=pdf", str(path), str(tmp)],
                capture_output=True, timeout=300,
            )
            if r.returncode == 0 and tmp.exists():
                raw = tmp.read_bytes()
                return parse_pdf_pymupdf(path, raw=raw)
        except Exception:
            pass
        finally:
            if tmp.exists():
                tmp.unlink()

    if _DJVUTXT:
        try:
            r = subprocess.run(
                [_DJVUTXT, str(path)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=300,
            )
            if r.returncode == 0 and r.stdout.strip():
                return [{"page": 1, "text": r.stdout}]
        except Exception:
            pass

    console.print(
        f"  [yellow]DJVU: no djvulibre CLI found for {path.name}. "
        "Install: conda install -c conda-forge djvulibre[/yellow]"
    )
    return []


def _get_jina_api_key() -> Optional[str]:
    key = os.environ.get("JINA_API_KEY")
    if not key:
        env = _HERE / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.startswith("JINA_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    return key if key else None

def embed_batch_jina_cloud(texts: list[str], task: str = "retrieval.passage") -> list[list[float]]:
    """Embed texts using Jina AI cloud API (jina-embeddings-v3, 1024-dim).
    Retries up to 6 times with exponential backoff on 429 rate-limit responses."""
    if not REQUESTS_OK:
        raise RuntimeError("requests not installed: pip install requests")
    api_key = _get_jina_api_key()
    if not api_key:
        raise RuntimeError("JINA_API_KEY not set in .env")
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), JINA_MAX_BATCH):
        batch = texts[i : i + JINA_MAX_BATCH]
        for attempt in range(6):
            resp = _requests.post(
                JINA_EMBED_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": JINA_EMBED_MODEL, "input": batch, "task": task},
                timeout=60,
            )
            if resp.status_code == 429:
                wait = (2 ** attempt) + random.random()
                console.print(f"[dim]Jina rate limit (batch {i//JINA_MAX_BATCH+1}), retry in {wait:.1f}s…[/dim]")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        data = resp.json()
        all_embeddings.extend(
            item["embedding"]
            for item in sorted(data["data"], key=lambda x: x["index"])
        )
    return all_embeddings

# ── Language detection + translation ─────────────────────────────────────────

_translation_cache: dict[str, Any] = {}

def detect_language(text: str) -> str:
    if not LANGDETECT_OK or not text.strip():
        return "en"
    try:
        return _langdetect(text[:500])
    except LangDetectException:
        return "en"

def translate_chunk(text: str, lang: str, device: str) -> str:
    """Translate non-English chunk to English. Returns original if lang=en or no model."""
    if lang == "en" or lang not in TRANSLATION_MODELS:
        return text
    if not TRANSFORMERS_OK:
        return text
    model_name = TRANSLATION_MODELS[lang]
    if model_name not in _translation_cache:
        try:
            _translation_cache[model_name] = hf_pipeline(
                "translation",
                model=model_name,
                device=0 if device == "cuda" else -1,
            )
        except Exception as exc:
            console.print(f"  [yellow]Translation model load failed ({model_name}): {exc}[/yellow]")
            _translation_cache[model_name] = None
    pipe = _translation_cache[model_name]
    if pipe is None:
        return text
    try:
        result = pipe(text[:512], max_length=512)
        return result[0]["translation_text"]
    except Exception:
        return text

# ── Chunking ──────────────────────────────────────────────────────────────────

# Sentence-end: punctuation followed by whitespace and uppercase/digit,
# but NOT preceded by known abbreviations (Mr, Dr, Vol, pp, ibid, etc.)
_ABBREV = re.compile(
    r'\b(Mr|Mrs|Ms|Dr|Prof|Rev|Sr|Jr|St|Vol|vol|No|no|pp|p|cf|ibid|et al|'
    r'op|cit|fig|Fig|ed|Ed|trans|Trans|repr|Repr|approx|dept|est|'
    r'[A-Z])$'
)
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-ZÀ-ɏ\d\"‘“])')

def split_sentences(text: str) -> list[str]:
    # Hard-split on paragraph boundaries first
    paragraphs = re.split(r'\n{2,}', text.strip())
    result: list[str] = []
    for para in paragraphs:
        raw = _SENTENCE_END.split(para.strip())
        # Re-join splits preceded by known abbreviations
        for part in raw:
            if result and _ABBREV.search(result[-1].rstrip().rstrip('.')):
                result[-1] = result[-1] + " " + part.strip()
            else:
                s = part.strip()
                if s:
                    result.append(s)
    return result

def make_chunks(
    pages: list[dict],
    source_file_id: str,
    lang: str,
    device: str,
) -> list[dict]:
    """
    Build a two-level chunk hierarchy:
      parent: up to PARENT_CHUNK_TOKENS tokens
      child:  up to CHILD_CHUNK_TOKENS tokens (always references parent)

    Returns flat list of chunk dicts (both parent and child nodes).
    """
    # Flatten all page text into sentence stream with page attribution
    sentences: list[tuple[int, str]] = []  # (page_number, sentence)
    for page in pages:
        raw = page["text"].strip()
        if not raw:
            continue
        for sent in split_sentences(raw):
            sentences.append((page["page"], sent))

    if not sentences:
        return []

    chunks: list[dict] = []
    passage_index = 0

    def _make_chunk(
        text_en: str, text_orig: str, page: int, parent_id: Optional[str], level: str
    ) -> dict:
        nonlocal passage_index
        c = {
            "id": str(uuid.uuid4()),
            "source_file_id": source_file_id,
            "passage_index": passage_index,
            "level": level,          # "parent" or "child"
            "parent_id": parent_id,
            "text": text_en,
            "text_original": text_orig if text_orig != text_en else None,
            "language_detected": lang,
            "page_number": page,
            "token_count": count_tokens(text_en),
        }
        passage_index += 1
        return c

    # Build parent chunks
    parent_buf: list[tuple[int, str]] = []
    parent_tokens = 0

    def flush_parent() -> Optional[dict]:
        if not parent_buf:
            return None
        raw = " ".join(s for _, s in parent_buf)
        translated = translate_chunk(raw, lang, device)
        page = parent_buf[0][0]
        return _make_chunk(translated, raw, page, None, "parent")

    for page_num, sent in sentences:
        sent_tokens = count_tokens(sent)
        if parent_tokens + sent_tokens > PARENT_CHUNK_TOKENS and parent_buf:
            parent = flush_parent()
            if parent and len(parent["text"]) >= MIN_CHUNK_CHARS:
                # Build child chunks from the parent text
                chunks.append(parent)
                _build_children(parent, chunks)
                # Overlap: keep last CHUNK_OVERLAP_TOKENS worth of sentences
                overlap: list[tuple[int, str]] = []
                overlap_tokens = 0
                for pg, s in reversed(parent_buf):
                    st = count_tokens(s)
                    if overlap_tokens + st > CHUNK_OVERLAP_TOKENS:
                        break
                    overlap.insert(0, (pg, s))
                    overlap_tokens += st
                parent_buf = overlap
                parent_tokens = overlap_tokens
        parent_buf.append((page_num, sent))
        parent_tokens += sent_tokens

    # Flush remaining
    parent = flush_parent()
    if parent and len(parent["text"]) >= MIN_CHUNK_CHARS:
        chunks.append(parent)
        _build_children(parent, chunks)

    return chunks

def _build_children(parent: dict, out: list[dict]) -> None:
    """Slice the parent text into child chunks of CHILD_CHUNK_TOKENS."""
    sents = split_sentences(parent["text"])
    buf: list[str] = []
    buf_tokens = 0
    child_index = 0

    def flush() -> None:
        nonlocal child_index
        if not buf:
            return
        text = " ".join(buf)
        if len(text) < MIN_CHUNK_CHARS:
            return
        out.append({
            "id": str(uuid.uuid4()),
            "source_file_id": parent["source_file_id"],
            "passage_index": parent["passage_index"] * 1000 + child_index,
            "level": "child",
            "parent_id": parent["id"],
            "text": text,
            "text_original": None,
            "language_detected": parent["language_detected"],
            "page_number": parent["page_number"],
            "token_count": count_tokens(text),
        })
        child_index += 1

    for sent in sents:
        t = count_tokens(sent)
        if buf_tokens + t > CHILD_CHUNK_TOKENS and buf:
            flush()
            # minimal overlap
            overlap = buf[-1:] if buf else []
            buf = overlap
            buf_tokens = count_tokens(overlap[0]) if overlap else 0
        buf.append(sent)
        buf_tokens += t
    flush()

# ── Embedding ─────────────────────────────────────────────────────────────────

_dense_model: Optional[Any] = None
_sparse_model: Optional[Any] = None

def get_dense_model(device: str) -> Optional[Any]:
    global _dense_model
    if _dense_model is None and FASTEMBED_OK:
        try:
            # Always include CPUExecutionProvider as fallback so CUDA failures don't hard-crash
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if device == "cuda"
                else ["CPUExecutionProvider"]
            )
            _dense_model = TextEmbedding(model_name=DENSE_MODEL, providers=providers)
        except Exception as exc:
            console.print(f"[yellow]Dense model load failed: {exc}. Trying BGE-M3 fallback.[/yellow]")
            try:
                _dense_model = TextEmbedding(model_name="BAAI/bge-large-en-v1.5")
            except Exception:
                _dense_model = None
    return _dense_model

def get_sparse_model() -> Optional[Any]:
    global _sparse_model
    if _sparse_model is None and FASTEMBED_OK:
        try:
            _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
        except Exception as exc:
            console.print(f"[yellow]Sparse model load failed: {exc}[/yellow]")
            _sparse_model = None
    return _sparse_model

def _warmup_models(device: str) -> None:
    """Embed a tiny batch to trigger ONNX/CUDA graph compilation before the main loop."""
    try:
        embed_batch(["GPU warmup"] * 4, device)
    except Exception:
        pass

def _embed_sparse_local(texts: list[str]) -> list[Optional[dict]]:
    """BM42 sparse embeddings (always local, even when using Jina cloud for dense)."""
    sparse_model = get_sparse_model()
    sparse: list[Optional[dict]] = [None] * len(texts)
    if sparse_model:
        try:
            for i, sv in enumerate(sparse_model.embed(texts)):
                sparse[i] = {"indices": sv.indices.tolist(), "values": sv.values.tolist()}
        except Exception as exc:
            console.print(f"[yellow]Sparse embed error: {exc}[/yellow]")
    return sparse


def embed_batch(
    texts: list[str], device: str, embed_via: str = "local"
) -> tuple[list[list[float]], list[Optional[dict]]]:
    """
    Returns (dense_vectors, sparse_vectors).
    embed_via='jina-cloud' calls Jina AI API for dense; sparse is always local BM42.
    """
    if embed_via == "jina-cloud":
        try:
            dense = embed_batch_jina_cloud(texts)
            sparse = _embed_sparse_local(texts)
            return dense, sparse
        except Exception as exc:
            console.print(f"[yellow]Jina cloud embed failed ({exc}) – falling back to local ONNX.[/yellow]")

    # Local ONNX path
    dense_model = get_dense_model(device)
    dense: list[list[float]] = []
    if dense_model:
        try:
            dense = [v.tolist() for v in dense_model.embed(texts)]
        except Exception as exc:
            console.print(f"[yellow]Dense embed OOM (batch={len(texts)}) — retrying one-by-one[/yellow]")
            for t in texts:
                try:
                    dense.append(next(iter(dense_model.embed([t]))).tolist())
                except Exception:
                    dense.append([0.0] * DENSE_DIM)
    else:
        dense = [[0.0] * DENSE_DIM] * len(texts)

    return dense, _embed_sparse_local(texts)

# ── Qdrant ─────────────────────────────────────────────────────────────────────

def get_qdrant(url: str, local_path: str = "./qdrant_data") -> Optional[Any]:
    if not QDRANT_OK:
        return None
    # Try remote server first
    try:
        client = QdrantClient(url=url, timeout=5)
        client.get_collections()
        console.print(f"[green]Connected to Qdrant server at {url}[/green]")
        return client
    except Exception:
        pass
    # Auto-fallback to local SQLite mode (no Docker required)
    try:
        Path(local_path).mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=local_path)
        console.print(
            f"[yellow]Qdrant server unavailable; using local storage at '{local_path}'[/yellow]\n"
            "[dim](Start Docker Desktop + qdrant/qdrant container for network access)[/dim]"
        )
        return client
    except Exception as exc:
        console.print(f"[red]Qdrant local mode also failed: {exc}[/red]")
        return None

def ensure_collection(client: Any, name: str, dense_dim: int, recreate: bool) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name in existing and recreate:
        client.delete_collection(name)
        existing.discard(name)
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config={"dense": qm.VectorParams(size=dense_dim, distance=qm.Distance.COSINE)},
            sparse_vectors_config={"sparse": qm.SparseVectorParams()},
        )
        console.print(f"[green]Created Qdrant collection '{name}'[/green]")

def upsert_chunks(
    client: Any,
    collection: str,
    chunks: list[dict],
    dense_vecs: list[list[float]],
    sparse_vecs: list[Optional[dict]],
    source_meta: dict,
    zot_meta: dict,
) -> int:
    points = []
    for chunk, dv, sv in zip(chunks, dense_vecs, sparse_vecs):
        vectors: dict = {"dense": dv}
        if sv:
            vectors["sparse"] = qm.SparseVector(
                indices=sv["indices"], values=sv["values"]
            )
        payload = {
            **{k: v for k, v in chunk.items() if k != "id"},
            **source_meta,
            **zot_meta,
        }
        points.append(
            qm.PointStruct(id=chunk["id"], vector=vectors, payload=payload)
        )
    if points:
        client.upsert(collection_name=collection, points=points, wait=True)
    return len(points)

# ── Per-file pipeline ──────────────────────────────────────────────────────────

EMBED_BATCH_SIZE = 16  # 16 fits GTX 1650 VRAM even for long (1000+ token) chunks

def process_file(
    path: Path,
    library_root: Path,
    state: dict,
    zot: dict[str, dict],
    client: Optional[Any],
    collection: str,
    device: str,
    dry_run: bool,
    embed_via: str = "local",
) -> dict:
    """
    Process one PDF or DJVU. Returns a status dict to merge into state["files"].
    """
    rel = str(path.relative_to(library_root))
    ext = path.suffix.lower()

    # ── Hash + parse, dispatched by format ────────────────────────────────────
    if ext == ".pdf":
        # Read once for both hash and parse (avoids double network I/O)
        try:
            raw = read_file_bytes(path)
        except OSError as exc:
            return {
                "path": rel, "hash": "", "status": "failed",
                "chunks_indexed": 0, "error": f"read error: {exc}",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        h = hash_bytes(raw)
        pages_fn = lambda: parse_pdf_from_bytes(raw, path)  # noqa: E731
    elif ext == ".djvu":
        # DJVU files are local; ddjvu needs a file path anyway
        try:
            h = file_hash(path)
        except OSError as exc:
            return {
                "path": rel, "hash": "", "status": "failed",
                "chunks_indexed": 0, "error": f"read error: {exc}",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        pages_fn = lambda: parse_djvu(path)  # noqa: E731
    else:
        return {
            "path": rel, "hash": "", "status": "failed",
            "chunks_indexed": 0, "error": f"unsupported extension: {ext}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    result: dict = {
        "path": rel,
        "hash": h,
        "status": "pending",
        "chunks_indexed": 0,
        "error": None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    if dry_run:
        result["status"] = "dry_run"
        return result

    # ── Parse ──────────────────────────────────────────────────────────────────
    pages = pages_fn()
    if not pages:
        result["status"] = "failed"
        result["error"] = "parse returned no pages"
        return result

    full_text = " ".join(p["text"] for p in pages)
    if len(full_text.strip()) < MIN_CHUNK_CHARS:
        result["status"] = "failed"
        result["error"] = "insufficient text extracted"
        return result

    # ── Language ───────────────────────────────────────────────────────────────
    lang = detect_language(full_text)

    # ── Chunk ──────────────────────────────────────────────────────────────────
    source_file_id = str(uuid.uuid5(uuid.NAMESPACE_URL, rel))
    chunks = make_chunks(pages, source_file_id, lang, device)
    if not chunks:
        result["status"] = "failed"
        result["error"] = "chunker produced no output"
        return result

    # ── Source metadata ────────────────────────────────────────────────────────
    zot_meta = zotero_metadata(path, zot)
    source_meta = {
        "source_file_id": source_file_id,
        "file_path": rel,
        "file_name": path.name,
        "page_count": len(pages),
        "checksum": h,
        **zot_meta,
    }

    # ── Embed + index ──────────────────────────────────────────────────────────
    # Only embed children: 256-token chunks give denser signal per retrieval slot.
    # Parents are discarded here; child text carries enough context for Claude.
    # Skipping parent embedding gives ~4x fewer ONNX forward passes per file.
    child_chunks = [c for c in chunks if c["level"] == "child"]
    index_chunks = child_chunks if child_chunks else chunks  # fallback if no children

    if client is None:
        result["status"] = "embedded_no_qdrant"
        result["chunks_indexed"] = len(index_chunks)
        return result

    total_upserted = 0
    texts = [c["text"] for c in index_chunks]
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch_chunks = index_chunks[i : i + EMBED_BATCH_SIZE]
        batch_texts = texts[i : i + EMBED_BATCH_SIZE]
        dense_vecs, sparse_vecs = embed_batch(batch_texts, device, embed_via=embed_via)
        n = upsert_chunks(client, collection, batch_chunks, dense_vecs, sparse_vecs, source_meta, zot_meta)
        total_upserted += n

    result["status"] = "indexed"
    result["chunks_indexed"] = total_upserted
    return result

# ── File collection ────────────────────────────────────────────────────────────

def collect_files(
    library_root: Path,
    state: dict,
    force: bool,
    limit: Optional[int],
    max_file_mb: Optional[float] = None,
) -> list[Path]:
    if not library_root.exists():
        console.print(f"[red]Library root not found: {library_root}[/red]")
        sys.exit(1)
    files = []
    skipped_large = 0
    max_bytes = int(max_file_mb * 1024 * 1024) if max_file_mb else None
    for ext in SUPPORTED_EXTENSIONS:
        for p in library_root.rglob(f"*{ext}"):
            if is_excluded(p):
                continue
            if max_bytes:
                try:
                    if p.stat().st_size > max_bytes:
                        skipped_large += 1
                        continue
                except OSError:
                    pass
            if not force:
                entry = state["files"].get(str(p.relative_to(library_root)))
                if entry and entry.get("status") == "indexed":
                    stored_hash = entry.get("hash", "")
                    try:
                        if stored_hash and file_hash(p) == stored_hash:
                            continue
                    except OSError:
                        continue
            files.append(p)
    files.sort()
    if skipped_large:
        console.print(f"[dim]Skipped {skipped_large} files exceeding {max_file_mb} MB size limit.[/dim]")
    if limit:
        files = files[:limit]
    return files

# ── Main ───────────────────────────────────────────────────────────────────────

def check_dependencies() -> list[str]:
    warnings = []
    if not PYMUPDF_OK:
        warnings.append("pymupdf missing - PDF parsing unavailable (pip install pymupdf)")
    if not FASTEMBED_OK:
        warnings.append("fastembed missing - embeddings will be zero vectors (pip install fastembed)")
    if not LANGDETECT_OK:
        warnings.append("langdetect missing - all chunks treated as English (pip install langdetect)")
    if not TIKTOKEN_OK:
        warnings.append("tiktoken missing - token counts are approximate (pip install tiktoken)")
    if not TRANSFORMERS_OK:
        warnings.append("transformers missing - FR/DE/LA translation disabled (pip install transformers)")
    if not QDRANT_OK:
        warnings.append("qdrant-client missing - cannot index vectors (pip install qdrant-client)")
    if not DOCLING_OK and not MARKER_OK:
        warnings.append("neither docling nor marker-pdf installed - scanned PDFs will use PyMuPDF only")
    if not DJVU_CLI_OK:
        warnings.append("djvulibre CLI not found - DJVU files will be skipped (conda install -c conda-forge djvulibre)")
    return warnings

def print_summary(state: dict) -> None:
    files = state.get("files", {})
    by_status: dict[str, int] = {}
    total_chunks = 0
    for entry in files.values():
        s = entry.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1
        total_chunks += entry.get("chunks_indexed", 0)

    t = Table(title="Ingestion Summary", show_header=True)
    t.add_column("Status")
    t.add_column("Files", justify="right")
    for status, count in sorted(by_status.items()):
        colour = "green" if status == "indexed" else "yellow" if "embed" in status else "red" if status == "failed" else "dim"
        t.add_row(f"[{colour}]{status}[/{colour}]", str(count))
    t.add_section()
    t.add_row("[bold]Total chunks[/bold]", f"[bold]{total_chunks:,}[/bold]")
    console.print(t)

def main() -> None:
    ap = argparse.ArgumentParser(description="Build the ALS library Qdrant index.")
    ap.add_argument("--dry-run", action="store_true", help="Scan files without parsing or indexing")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N files (for testing)")
    ap.add_argument("--force-reindex", action="store_true", help="Ignore dirty state; re-index everything")
    ap.add_argument("--library-root", type=Path, default=LIBRARY_ROOT)
    ap.add_argument("--zotero-export", type=Path, default=None, help="Path to BetterBibTeX JSON export")
    ap.add_argument("--qdrant-url", default=QDRANT_URL)
    ap.add_argument("--qdrant-path", default=str(_HERE / "qdrant_data"),
                    help="Local Qdrant storage path (used when server is unavailable)")
    ap.add_argument("--collection", default=COLLECTION_NAME)
    ap.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    ap.add_argument("--recreate-collection", action="store_true", help="Drop and recreate the Qdrant collection")
    ap.add_argument("--max-file-mb", type=float, default=None,
                    help="Skip files larger than this many megabytes (useful to skip huge outliers during testing)")
    ap.add_argument("--embed-via", choices=["local", "jina-cloud"], default="local",
                    help="Embedding backend: 'local' uses Jina v3 ONNX on CPU/GPU; "
                         "'jina-cloud' uses Jina AI API (requires JINA_API_KEY in .env)")
    args = ap.parse_args()

    # ── Dependency check ───────────────────────────────────────────────────────
    warnings = check_dependencies()
    for w in warnings:
        console.print(f"[yellow]WARNING: {w}[/yellow]")
    if not PYMUPDF_OK and not DOCLING_OK and not MARKER_OK:
        console.print("[red]No PDF parser available. Install pymupdf: pip install pymupdf[/red]")
        sys.exit(1)

    state = load_state()
    zot = load_zotero_index(args.zotero_export)
    if zot:
        console.print(f"[green]Loaded Zotero index: {len(zot)} items[/green]")
    else:
        console.print("[dim]No Zotero export provided; metadata inferred from filenames.[/dim]")

    # ── Qdrant setup ───────────────────────────────────────────────────────────
    client: Optional[Any] = None
    if not args.dry_run:
        client = get_qdrant(args.qdrant_url, args.qdrant_path)
        if client:
            ensure_collection(client, args.collection, DENSE_DIM, args.recreate_collection)

    # ── Collect files ──────────────────────────────────────────────────────────
    files = collect_files(args.library_root, state, args.force_reindex, args.limit, args.max_file_mb)
    already_done = sum(
        1 for e in state.get("files", {}).values() if e.get("status") == "indexed"
    )
    console.print(
        f"\n[bold]Files to process:[/bold] {len(files):,}  "
        f"[dim](already indexed: {already_done:,})[/dim]"
    )

    if args.dry_run:
        for f in files:
            try:
                rel = f.relative_to(args.library_root)
            except ValueError:
                rel = f
            console.print(f"  [dim]DRY RUN:[/dim] {rel}")
        console.print(f"\n[bold]{len(files)}[/bold] files would be processed.")
        return

    if not files:
        console.print("[green]Nothing to do — all files up to date.[/green]")
        print_summary(state)
        return

    # ── Pre-load embedding models (avoid per-file cold starts) ─────────────────
    if not args.dry_run and FASTEMBED_OK:
        console.print("[dim]Loading embedding models...[/dim]")
        if args.embed_via == "local":
            t0 = time.time()
            console.print(f"[dim]  Loading dense model ({DENSE_MODEL}) on {args.device}...[/dim]")
            get_dense_model(args.device)
            console.print(f"[dim]  Dense model loaded in {time.time()-t0:.1f}s[/dim]")
        t1 = time.time()
        console.print("[dim]  Loading sparse model (BM42)...[/dim]")
        get_sparse_model()
        console.print(f"[dim]  Sparse model loaded in {time.time()-t1:.1f}s[/dim]")
        console.print("[dim]  Running GPU warmup batch...[/dim]")
        _warmup_models(args.device)
        console.print("[dim]  Warmup complete.[/dim]")

    # ── Main loop ──────────────────────────────────────────────────────────────
    ok = fail = 0
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing", total=len(files))
        for path in files:
            try:
                rel = path.relative_to(args.library_root)
            except ValueError:
                rel = path
            progress.update(task, description=f"[cyan]{rel.name[:50]}[/cyan]")
            t0 = time.time()
            try:
                result = process_file(
                    path=path,
                    library_root=args.library_root,
                    state=state,
                    zot=zot,
                    client=client,
                    collection=args.collection,
                    device=args.device,
                    dry_run=False,
                    embed_via=args.embed_via,
                )
                elapsed_s = time.time() - t0
                state.setdefault("files", {})[str(rel)] = result
                if result["status"] in ("indexed", "embedded_no_qdrant"):
                    ok += 1
                    chunks = result.get("chunks_indexed", 0)
                    rate = chunks / elapsed_s if elapsed_s > 0 else 0
                    console.print(
                        f"  [green]OK[/green] {rel.name[:50]} "
                        f"[dim]({chunks} chunks, {elapsed_s:.0f}s, {rate:.1f} c/s)[/dim]"
                    )
                else:
                    fail += 1
                    console.print(
                        f"  [red]FAIL[/red] {rel.name}: {result.get('error', 'unknown')}"
                    )
            except Exception as exc:
                fail += 1
                console.print(f"  [red]EXCEPTION[/red] {rel.name}: {exc}")
                state.setdefault("files", {})[str(rel)] = {
                    "path": str(rel),
                    "status": "failed",
                    "error": str(exc),
                    "chunks_indexed": 0,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            finally:
                save_state(state)
                progress.advance(task)

    console.print(f"\n[bold green]{ok}[/bold green] indexed, [bold red]{fail}[/bold red] failed")
    print_summary(state)


if __name__ == "__main__":
    main()
