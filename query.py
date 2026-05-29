#!/usr/bin/env python3
"""
query.py — Query engine for the Auguste Laurent Society library index.

Pipeline per query:
  1. HyDE: Claude Haiku generates a hypothetical answer paragraph
  2. Embed the hypothesis (Jina v3 dense + BM42 sparse)
  3. Hybrid retrieval from Qdrant (dense ANN + sparse BM42)
  4. BGE-Reranker-v2-M3 cross-encoder (if available) -> top 10
  5. LLMLingua-2 context compression (if available) -> ~75% token reduction
  6. "Found in the Middle" ordering (best first, best last)
  7. Claude claude-sonnet-4-6 generation with citations
  8. MCP server mode: expose query_library(question) tool

Usage (interactive):
    python query.py

Usage (MCP server for Claude Desktop / Cursor / VS Code):
    python query.py --mcp

Single query:
    python query.py --ask "Who collaborated with Lavoisier on the 1789 nomenclature?"

Requirements:
    pip install qdrant-client fastembed anthropic rich
    # Optional reranking:
    pip install FlagEmbedding          # BGE-Reranker-v2-M3
    pip install llmlingua              # LLMLingua-2 compression
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import mimetypes
import os
import subprocess
import sys
import time
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Optional

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message="Failed to obtain server version")
warnings.filterwarnings("ignore", message=".*symlinks.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="fastembed")

# Add NVIDIA CUDA 12 DLL directories so onnxruntime-gpu finds cublasLt64_12.dll
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

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

console = Console()

# ── Optional imports ──────────────────────────────────────────────────────────

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
    import anthropic as _anthropic
    ANTHROPIC_OK = True
except ImportError:
    ANTHROPIC_OK = False

try:
    import openai as _openai
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False

try:
    from FlagEmbedding import FlagReranker
    BGE_RERANKER_OK = True
except ImportError:
    BGE_RERANKER_OK = False

try:
    from llmlingua import PromptCompressor
    LLMLINGUA_OK = True
except ImportError:
    LLMLINGUA_OK = False

try:
    from mcp.server.fastmcp import FastMCP
    MCP_OK = True
except ImportError:
    MCP_OK = False

# ── Configuration ─────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent

QDRANT_URL = "http://localhost:6333"
QDRANT_LOCAL_PATH = str(_HERE / "qdrant_data")
COLLECTION_NAME = "als_library"
TIMELINE_COLLECTION = "sacred_timeline"
TIMELINE_TOP_K = 5          # max person records retrieved per query
DENSE_MODEL = "jinaai/jina-embeddings-v3"
SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
DENSE_DIM = 1024

RETRIEVE_DENSE_TOP_K = 30
RETRIEVE_SPARSE_TOP_K = 20
RERANK_TOP_K = 10        # after BGE cross-encoder
CONTEXT_TOP_K = 5        # final passages sent to Claude

CLAUDE_GENERATION_MODEL = "claude-sonnet-4-6"
CLAUDE_HYDE_MODEL = "claude-haiku-4-5-20251001"
OPENAI_GENERATION_MODEL = "gpt-4o"
OPENAI_HYDE_MODEL = "gpt-4o-mini"
BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Free-tier models for parallel HyDE and generation.
# Verified live as of 2026-05-27 via /api/v1/models.
# Each model has its own daily request limit; N models = N × daily_limit capacity.
OPENROUTER_HYDE_MODELS = [
    # Skip hermes-3-405b here to preserve its daily quota for generation/synthesis
    "nvidia/nemotron-3-super-120b-a12b:free",       # 120B, 1M ctx
    "meta-llama/llama-3.3-70b-instruct:free",       # 70B, solid instruction-following
    "openai/gpt-oss-120b:free",                     # 120B OSS from OpenAI
]
OPENROUTER_GEN_MODELS = [
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openai/gpt-oss-120b:free",
]
# Synthesis model — consolidates N generation answers into one scholarly response
OPENROUTER_SYNTH_MODEL = "nousresearch/hermes-3-llama-3.1-405b:free"

SYSTEM_PROMPT = """You are a scholarly research assistant specializing in the history of science, \
mathematics, and natural philosophy. You have access to a curated library of primary and secondary \
sources from the Auguste Laurent Society collection.

Answer questions with precision, cite your sources explicitly (author, title, page where available), \
and distinguish clearly between what the sources say and your own synthesis. \
If the evidence is thin or contradictory, say so."""

# ── Singleton models ──────────────────────────────────────────────────────────

_dense_model: Optional[Any] = None
_sparse_model: Optional[Any] = None
_reranker: Optional[Any] = None
_compressor: Optional[Any] = None
_anthropic_client: Optional[Any] = None
_anthropic_init_tried: bool = False
_openai_client: Optional[Any] = None
_openai_init_tried: bool = False
_openrouter_client: Optional[Any] = None
_openrouter_init_tried: bool = False


def get_dense_model() -> Optional[Any]:
    global _dense_model
    if _dense_model is None and FASTEMBED_OK:
        try:
            _dense_model = TextEmbedding(model_name=DENSE_MODEL)
        except Exception as exc:
            console.print(f"[yellow]Dense model failed: {exc}[/yellow]")
    return _dense_model


def get_sparse_model() -> Optional[Any]:
    global _sparse_model
    if _sparse_model is None and FASTEMBED_OK:
        try:
            _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
        except Exception as exc:
            console.print(f"[yellow]Sparse model failed: {exc}[/yellow]")
    return _sparse_model


def get_reranker() -> Optional[Any]:
    global _reranker
    if _reranker is None and BGE_RERANKER_OK:
        try:
            _reranker = FlagReranker(BGE_RERANKER_MODEL, use_fp16=True)
        except Exception as exc:
            console.print(f"[yellow]BGE reranker failed to load: {exc}[/yellow]")
    return _reranker


def get_compressor() -> Optional[Any]:
    global _compressor
    if _compressor is None and LLMLINGUA_OK:
        try:
            _compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True,
            )
        except Exception as exc:
            console.print(f"[yellow]LLMLingua-2 failed to load: {exc}[/yellow]")
    return _compressor


def _load_env_key(var: str) -> Optional[str]:
    """Read an API key from environment or script-relative .env file."""
    key = os.environ.get(var)
    if not key:
        env = _HERE / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.startswith(f"{var}="):
                    key = line.split("=", 1)[1].strip()
                    break
    return key if key and not key.startswith("your_") else None


def get_anthropic() -> Optional[Any]:
    global _anthropic_client, _anthropic_init_tried
    if _anthropic_client is None and ANTHROPIC_OK and not _anthropic_init_tried:
        _anthropic_init_tried = True
        api_key = _load_env_key("ANTHROPIC_API_KEY")
        if api_key:
            _anthropic_client = _anthropic.Anthropic(api_key=api_key)
        else:
            Console(stderr=True).print("[yellow]ANTHROPIC_API_KEY not set - Claude generation disabled[/yellow]")
    return _anthropic_client


def get_openai() -> Optional[Any]:
    global _openai_client, _openai_init_tried
    if _openai_client is None and OPENAI_OK and not _openai_init_tried:
        _openai_init_tried = True
        api_key = _load_env_key("OPENAI_API_KEY")
        if api_key:
            _openai_client = _openai.OpenAI(api_key=api_key)
        else:
            Console(stderr=True).print("[yellow]OPENAI_API_KEY not set - OpenAI generation disabled[/yellow]")
    return _openai_client


def get_openrouter() -> Optional[Any]:
    global _openrouter_client, _openrouter_init_tried
    if _openrouter_client is None and OPENAI_OK and not _openrouter_init_tried:
        _openrouter_init_tried = True
        api_key = _load_env_key("OPENROUTER_API_KEY")
        if api_key:
            _openrouter_client = _openai.OpenAI(
                api_key=api_key,
                base_url=OPENROUTER_BASE_URL,
            )
        else:
            Console(stderr=True).print("[yellow]OPENROUTER_API_KEY not set - OpenRouter ensemble disabled[/yellow]")
    return _openrouter_client


def _openrouter_call(
    model: str,
    messages: list[dict],
    max_tokens: int = 400,
    timeout: float = 45.0,
) -> tuple[str, Optional[str]]:
    """Single call to one OpenRouter model. Returns (model_short_name, text_or_None)."""
    short = model.split("/")[-1].replace(":free", "")
    client = get_openrouter()
    if not client:
        return short, None
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            timeout=timeout,
        )
        text = resp.choices[0].message.content
        return short, text.strip() if text else None
    except Exception as exc:
        console.print(f"[dim]{short}: {exc}[/dim]")
        return short, None


# ── Qdrant ────────────────────────────────────────────────────────────────────

_qdrant_reachable_cache: tuple = (False, 0.0)  # (result, timestamp)
_QDRANT_REACHABLE_TTL = 5.0  # seconds between socket probes


def _qdrant_server_reachable() -> bool:
    """Fast socket-level check (0.5 s) before attempting full HTTP client. Cached for 5 s."""
    global _qdrant_reachable_cache
    result, ts = _qdrant_reachable_cache
    if time.monotonic() - ts < _QDRANT_REACHABLE_TTL:
        return result
    import socket
    try:
        with socket.create_connection(("localhost", 6333), timeout=0.5):
            result = True
    except OSError:
        result = False
    _qdrant_reachable_cache = (result, time.monotonic())
    return result


def _qdrant_local_locked() -> bool:
    """Non-blocking portalocker check — returns True immediately if another process holds the lock."""
    lock_path = Path(QDRANT_LOCAL_PATH) / ".lock"
    if not lock_path.exists():
        return False
    try:
        import portalocker
        with open(lock_path, "r") as f:
            portalocker.lock(f, portalocker.LOCK_EX | portalocker.LOCK_NB)
            portalocker.unlock(f)
        return False
    except Exception:
        return True


def get_qdrant() -> Optional[Any]:
    if not QDRANT_OK:
        return None
    if _qdrant_server_reachable():
        try:
            c = QdrantClient(url=QDRANT_URL, timeout=5)
            c.get_collections()
            return c
        except Exception:
            pass
    if not Path(QDRANT_LOCAL_PATH).exists():
        console.print("[red]No Qdrant index found. Run build_index.py first.[/red]")
        return None
    if _qdrant_local_locked():
        console.print(
            "[yellow]Qdrant index is locked by the indexer. "
            "Queries are unavailable while build_index.py runs. "
            "Start Docker Qdrant for concurrent access.[/yellow]"
        )
        return None
    try:
        return QdrantClient(path=QDRANT_LOCAL_PATH)
    except RuntimeError as exc:
        if "already accessed" in str(exc):
            console.print(
                "[yellow]Qdrant index is locked by the indexer. "
                "Queries are unavailable while build_index.py runs. "
                "Start Docker Qdrant for concurrent access.[/yellow]"
            )
        else:
            console.print(f"[red]Qdrant local open failed: {exc}[/red]")
    except Exception as exc:
        console.print(f"[red]Qdrant local open failed: {exc}[/red]")
    return None


def collection_exists(client: Any) -> bool:
    try:
        cols = {c.name for c in client.get_collections().collections}
        return COLLECTION_NAME in cols
    except Exception:
        return False


def timeline_collection_exists(client: Any) -> bool:
    try:
        cols = {c.name for c in client.get_collections().collections}
        return TIMELINE_COLLECTION in cols
    except Exception:
        return False


# ── Step 1: HyDE ─────────────────────────────────────────────────────────────

_HYDE_SYSTEM = (
    "You are generating a dense retrieval hypothesis. Write a single paragraph "
    "(100-150 words) that looks like a passage from a scholarly book or article "
    "that directly answers the following question. Do not hedge; write as if you "
    "are the source text itself."
)

# ── OpenRouter multi-model ensemble ──────────────────────────────────────────

def hyde_expand_openrouter(question: str) -> list[str]:
    """
    Parallel HyDE across OPENROUTER_HYDE_MODELS (60s per model timeout).
    Returns list of hypothetical passages (one per model that responded).
    Each becomes a separate retrieval vector → broader candidate pool.
    """
    messages = [
        {"role": "system", "content": _HYDE_SYSTEM},
        {"role": "user", "content": question},
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(OPENROUTER_HYDE_MODELS)) as ex:
        futures = [ex.submit(_openrouter_call, m, messages, 350, 60.0) for m in OPENROUTER_HYDE_MODELS]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    hypotheses = [text for _, text in results if text]
    return hypotheses if hypotheses else [question]


def retrieve_multi_hypothesis(
    qdrant_client: Any,
    hypotheses: list[str],
) -> list[dict]:
    """
    Embed each hypothesis separately, retrieve candidates for each,
    then fuse with summed RRF scores across all hypothesis retrievals.
    More hypotheses → broader semantic coverage of the collection.
    """
    pooled: dict[str, dict] = {}
    for hyp in hypotheses:
        dense_vec, sparse_vec = embed_query(hyp)
        candidates = retrieve(qdrant_client, dense_vec, sparse_vec)
        for c in candidates:
            pid = c["id"]
            if pid not in pooled:
                pooled[pid] = dict(c)
                pooled[pid]["rrf_score"] = 0.0
            pooled[pid]["rrf_score"] += c.get("rrf_score", 0.0)
    return sorted(pooled.values(), key=lambda x: x["rrf_score"], reverse=True)


def generate_answer_openrouter(question: str, candidates: list[dict]) -> str:
    """
    1. Assemble context from top candidates (same as single-model path).
    2. Query all OPENROUTER_GEN_MODELS in parallel with that context.
    3. Feed all responses to OPENROUTER_SYNTH_MODEL for scholarly synthesis.
    """
    top = candidates[:CONTEXT_TOP_K]
    raw_passages = [c["payload"].get("text", "") for c in top]
    compressed_passages = compress_context(raw_passages, question)
    ordered_passages = found_in_middle_order(compressed_passages)

    reordered = []
    for i, text in enumerate(ordered_passages):
        idx = i if i < len(top) else len(top) - 1
        c = dict(top[idx])
        c["payload"] = dict(c["payload"])
        c["payload"]["text"] = text
        reordered.append(c)

    context_blocks = [format_passage(c, i) for i, c in enumerate(reordered)]
    context = "\n\n---\n\n".join(context_blocks)
    user_content = (
        f"Using ONLY the sources provided below, answer the following question. "
        f"Cite each source by its [Source N] label when drawing from it.\n\n"
        f"Question: {question}\n\n"
        f"Sources:\n\n{context}"
    )
    gen_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # Phase 1: parallel generation across all models (90s each)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(OPENROUTER_GEN_MODELS)) as ex:
        futures = [ex.submit(_openrouter_call, m, gen_messages, 1200, 90.0) for m in OPENROUTER_GEN_MODELS]
        raw_answers: dict[str, str] = {}
        for f in concurrent.futures.as_completed(futures):
            short, text = f.result()
            if text:
                raw_answers[short] = text

    if not raw_answers:
        return (
            "All OpenRouter models failed or timed out. Top retrieved passage:\n\n"
            + (raw_passages[0] if raw_passages else "(none)")
        )

    # If only one model responded, return it directly
    if len(raw_answers) == 1:
        return next(iter(raw_answers.values()))

    # Phase 2: synthesis — try models in priority order until one succeeds
    answers_block = "\n\n".join(
        f"[{name}]:\n{text}" for name, text in raw_answers.items()
    )
    synth_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"The following are answers from {len(raw_answers)} AI models to the question:\n"
                f"\"{question}\"\n\n"
                f"{answers_block}\n\n"
                "Synthesize these into one comprehensive, well-cited scholarly answer. "
                "Preserve citation labels ([Source N]) from the original answers. "
                "Where models agree, present that as confident consensus. "
                "Where they disagree, note the disagreement explicitly."
            ),
        },
    ]
    # Try synthesis with each model in priority order until one succeeds
    synth_priority = [OPENROUTER_SYNTH_MODEL] + [
        m for m in OPENROUTER_GEN_MODELS if m != OPENROUTER_SYNTH_MODEL
    ]
    for synth_model in synth_priority:
        _, synth_text = _openrouter_call(synth_model, synth_messages, 1500, 90.0)
        if synth_text:
            return synth_text

    # All synthesis models failed — return concatenated answers as fallback
    return "\n\n---\n\n".join(
        f"**{name}:**\n{text}" for name, text in raw_answers.items()
    )


def hyde_expand(question: str, llm: str = "claude") -> str:
    """
    Generate a hypothetical answer paragraph for the question.
    Embedding this instead of the raw question dramatically improves recall.
    Falls back to the original question if the LLM is unavailable.
    """
    if llm == "openai":
        client = get_openai()
        if not client:
            return question
        try:
            resp = client.chat.completions.create(
                model=OPENAI_HYDE_MODEL,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": _HYDE_SYSTEM},
                    {"role": "user", "content": question},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            console.print(f"[dim]HyDE (OpenAI) failed ({exc}), using raw query[/dim]")
            return question

    # Claude (default)
    client = get_anthropic()
    if not client:
        return question
    try:
        msg = client.messages.create(
            model=CLAUDE_HYDE_MODEL,
            max_tokens=300,
            system=_HYDE_SYSTEM,
            messages=[{"role": "user", "content": question}],
        )
        return msg.content[0].text.strip()
    except Exception as exc:
        console.print(f"[dim]HyDE failed ({exc}), using raw query[/dim]")
        return question


# ── Step 2: Embed query ───────────────────────────────────────────────────────

def embed_query(text: str) -> tuple[list[float], Optional[dict]]:
    """Returns (dense_vector, sparse_vector)."""
    dense_vec: list[float] = [0.0] * DENSE_DIM
    sparse_vec: Optional[dict] = None

    dense_model = get_dense_model()
    if dense_model:
        try:
            dense_vec = list(next(iter(dense_model.embed([text]))))
        except Exception as exc:
            console.print(f"[yellow]Dense query embed failed: {exc}[/yellow]")

    sparse_model = get_sparse_model()
    if sparse_model:
        try:
            sv = next(iter(sparse_model.embed([text])))
            sparse_vec = {"indices": sv.indices.tolist(), "values": sv.values.tolist()}
        except Exception as exc:
            console.print(f"[yellow]Sparse query embed failed: {exc}[/yellow]")

    return dense_vec, sparse_vec


# ── Step 3: Retrieve ──────────────────────────────────────────────────────────

def retrieve(
    client: Any,
    dense_vec: list[float],
    sparse_vec: Optional[dict],
) -> list[dict]:
    """
    Hybrid retrieval: dense ANN + sparse BM42, fused with Reciprocal Rank Fusion.
    Returns list of candidate dicts ordered by RRF score.
    """
    candidates: dict[str, dict] = {}

    # Dense retrieval
    try:
        dense_results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=dense_vec,
            using="dense",
            limit=RETRIEVE_DENSE_TOP_K,
            with_payload=True,
        )
        for rank, pt in enumerate(dense_results.points):
            pid = str(pt.id)
            candidates.setdefault(pid, {"id": pid, "payload": pt.payload, "dense_rank": None, "sparse_rank": None})
            candidates[pid]["dense_rank"] = rank
            candidates[pid]["dense_score"] = pt.score
    except Exception as exc:
        console.print(f"[yellow]Dense retrieval failed: {exc}[/yellow]")

    # Sparse retrieval
    if sparse_vec:
        try:
            sparse_results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=qm.SparseVector(
                    indices=sparse_vec["indices"],
                    values=sparse_vec["values"],
                ),
                using="sparse",
                limit=RETRIEVE_SPARSE_TOP_K,
                with_payload=True,
            )
            for rank, pt in enumerate(sparse_results.points):
                pid = str(pt.id)
                candidates.setdefault(pid, {"id": pid, "payload": pt.payload, "dense_rank": None, "sparse_rank": None})
                candidates[pid]["sparse_rank"] = rank
        except Exception as exc:
            console.print(f"[yellow]Sparse retrieval failed: {exc}[/yellow]")

    # Timeline collection (person records) — namespaced "tl:{id}" to avoid collision
    if timeline_collection_exists(client):
        try:
            tl_results = client.query_points(
                collection_name=TIMELINE_COLLECTION,
                query=dense_vec,
                using="dense",
                limit=TIMELINE_TOP_K,
                with_payload=True,
            )
            for rank, pt in enumerate(tl_results.points):
                pid = f"tl:{pt.id}"
                candidates[pid] = {
                    "id": pid,
                    "payload": pt.payload,
                    "dense_rank": rank,
                    "sparse_rank": None,
                    "rrf_score": 0.0,
                }
        except Exception as exc:
            console.print(f"[yellow]Timeline retrieval failed: {exc}[/yellow]")

    # Reciprocal Rank Fusion (k=60 is standard)
    k = 60
    for cand in candidates.values():
        rrf = 0.0
        if cand["dense_rank"] is not None:
            rrf += 1.0 / (k + cand["dense_rank"] + 1)
        if cand["sparse_rank"] is not None:
            rrf += 1.0 / (k + cand["sparse_rank"] + 1)
        cand["rrf_score"] = rrf

    ranked = sorted(candidates.values(), key=lambda x: x["rrf_score"], reverse=True)
    return ranked


# ── Step 4: BGE cross-encoder rerank ─────────────────────────────────────────

def rerank(question: str, candidates: list[dict]) -> list[dict]:
    """
    Cross-encoder reranking with BGE-Reranker-v2-M3.
    Falls back to RRF order if reranker is unavailable.
    """
    if not candidates:
        return candidates

    reranker = get_reranker()
    if not reranker:
        return candidates[:RERANK_TOP_K]

    pairs = [[question, c["payload"].get("text", "")] for c in candidates]
    try:
        scores = reranker.compute_score(pairs, normalize=True)
        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = float(score)
        candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    except Exception as exc:
        console.print(f"[yellow]Reranker failed: {exc}[/yellow]")

    return candidates[:RERANK_TOP_K]


# ── Step 5: LLMLingua-2 context compression ───────────────────────────────────

def compress_context(passages: list[str], question: str) -> list[str]:
    """
    Token-level compression of each passage using LLMLingua-2.
    Targets 4x compression ratio with minimal quality loss.
    Falls back to original passages if compressor unavailable.
    """
    compressor = get_compressor()
    if not compressor:
        return passages

    compressed = []
    for passage in passages:
        if len(passage) < 200:   # too short to compress
            compressed.append(passage)
            continue
        try:
            result = compressor.compress_prompt(
                passage,
                instruction=question,
                rate=0.25,           # keep 25% of tokens
                force_tokens=["\n"],
            )
            compressed.append(result["compressed_prompt"])
        except Exception:
            compressed.append(passage)
    return compressed


# ── Step 6: "Found in the Middle" ordering ────────────────────────────────────

def found_in_middle_order(passages: list[str]) -> list[str]:
    """
    Place the most relevant passage first, second-most relevant last,
    and lower-ranked passages in the middle. Mitigates the "Lost in the
    Middle" attention bias in LLMs (Liu et al. 2023, Hsieh et al. 2024).
    """
    if len(passages) <= 2:
        return passages
    best = passages[0]
    second_best = passages[-1]
    middle = passages[1:-1]
    return [best] + middle + [second_best]


# ── Step 7: Build context + generate ─────────────────────────────────────────

def format_passage(cand: dict, index: int) -> str:
    p = cand["payload"]
    if p.get("source_type") == "sacred_timeline":
        name = p.get("name", "Unknown")
        birth = p.get("birth_year")
        death = p.get("death_year")
        era = p.get("calculus_number")
        dates = f" ({birth}–{death})" if birth and death else f" ({birth})" if birth else ""
        era_note = f" [Era {era}]" if era is not None else ""
        citation = f"Sacred Timeline — {name}{dates}{era_note}"
        return f"[Source {index + 1}: {citation}]\n{p.get('text', '')}"

    source = p.get("title") or p.get("file_name", "Unknown source")
    author = p.get("author", "")
    year = p.get("year", "")
    page = p.get("page_number", "")

    citation_parts = [source]
    if author:
        citation_parts.insert(0, str(author))
    if year:
        citation_parts.append(str(year))
    if page:
        citation_parts.append(f"p. {page}")
    citation = ", ".join(citation_parts)

    return f"[Source {index + 1}: {citation}]\n{p.get('text', '')}"


def generate_answer(question: str, candidates: list[dict], llm: str = "claude") -> str:
    """
    Assemble context from top candidates, compress, order, and call the selected LLM.
    """
    top = candidates[:CONTEXT_TOP_K]
    raw_passages = [c["payload"].get("text", "") for c in top]
    compressed_passages = compress_context(raw_passages, question)
    ordered_passages = found_in_middle_order(compressed_passages)

    # Rebuild candidates in new order, preserving citation metadata
    reordered = []
    for i, text in enumerate(ordered_passages):
        idx = i if i < len(top) else len(top) - 1
        c = dict(top[idx])
        c["payload"] = dict(c["payload"])
        c["payload"]["text"] = text
        reordered.append(c)

    context_blocks = [format_passage(c, i) for i, c in enumerate(reordered)]
    context = "\n\n---\n\n".join(context_blocks)
    user_content = (
        f"Using ONLY the sources provided below, answer the following question. "
        f"Cite each source by its [Source N] label when drawing from it.\n\n"
        f"Question: {question}\n\n"
        f"Sources:\n\n{context}"
    )

    if llm == "openai":
        oa = get_openai()
        if not oa:
            return (
                "OpenAI API unavailable. Top retrieved passages:\n\n"
                + "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(raw_passages[:3]))
            )
        try:
            resp = oa.chat.completions.create(
                model=OPENAI_GENERATION_MODEL,
                max_tokens=1500,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            return f"OpenAI generation error: {exc}\n\nTop passage: {raw_passages[0] if raw_passages else '(none)'}"

    # Claude (default)
    client = get_anthropic()
    if not client:
        return (
            "Claude API unavailable. Top retrieved passages:\n\n"
            + "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(raw_passages[:3]))
        )
    try:
        response = client.messages.create(
            model=CLAUDE_GENERATION_MODEL,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        return response.content[0].text.strip()
    except Exception as exc:
        return f"Generation error: {exc}\n\nTop passage: {raw_passages[0] if raw_passages else '(none)'}"


# ── Full query pipeline ───────────────────────────────────────────────────────

def query(question: str, verbose: bool = False, llm: str = "claude") -> dict:
    """
    End-to-end query: HyDE -> embed -> retrieve -> rerank -> compress -> generate.
    llm='openrouter' uses parallel free models for HyDE + generation + synthesis.
    Returns dict with 'answer', 'sources', 'candidates'.
    """
    qdrant_client = get_qdrant()
    if not qdrant_client or not collection_exists(qdrant_client):
        locked = qdrant_client is None and Path(QDRANT_LOCAL_PATH).exists()
        return {
            "answer": (
                "The index is currently locked by the indexer — try again once "
                "build_index.py finishes, or start a Docker Qdrant container for "
                "concurrent access."
                if locked else
                "No index found. Run build_index.py to index the library first."
            ),
            "sources": [],
            "candidates": [],
        }

    if llm == "openrouter":
        # ── Multi-model ensemble path ───────────────────────────────────────
        if verbose:
            console.print(f"[dim]Step 1: Multi-model HyDE ({len(OPENROUTER_HYDE_MODELS)} parallel models)...[/dim]")
        hypotheses = hyde_expand_openrouter(question)
        if verbose:
            console.print(f"[dim]  {len(hypotheses)} hypotheses generated[/dim]")
            for i, h in enumerate(hypotheses, 1):
                console.print(Panel(h[:300], title=f"[dim]HyDE {i}[/dim]", border_style="dim"))

        if verbose:
            console.print(f"[dim]Step 2-3: Multi-hypothesis embedding + pooled retrieval...[/dim]")
        candidates = retrieve_multi_hypothesis(qdrant_client, hypotheses)
        if verbose:
            console.print(f"[dim]  {len(candidates)} candidates after fusion[/dim]")

        if verbose:
            console.print("[dim]Step 4: Reranking...[/dim]")
        candidates = rerank(question, candidates)

        if verbose:
            console.print(f"[dim]Step 5-7: Multi-model generation ({len(OPENROUTER_GEN_MODELS)} models) + synthesis...[/dim]")
        answer = generate_answer_openrouter(question, candidates)

    else:
        # ── Single-model path (claude / openai) ────────────────────────────
        if verbose:
            console.print(f"[dim]Step 1: HyDE expansion (llm={llm})...[/dim]")
        hypothesis = hyde_expand(question, llm=llm)
        if verbose and hypothesis != question:
            console.print(Panel(hypothesis, title="[dim]HyDE hypothesis[/dim]", border_style="dim"))

        if verbose:
            console.print("[dim]Step 2: Embedding query...[/dim]")
        dense_vec, sparse_vec = embed_query(hypothesis)

        if verbose:
            console.print("[dim]Step 3: Hybrid retrieval (dense + sparse)...[/dim]")
        candidates = retrieve(qdrant_client, dense_vec, sparse_vec)
        if verbose:
            console.print(f"[dim]  Retrieved {len(candidates)} candidates[/dim]")

        if verbose:
            console.print("[dim]Step 4: Reranking...[/dim]")
        candidates = rerank(question, candidates)

        if verbose:
            console.print("[dim]Step 5-7: Compress + order + generate...[/dim]")
        answer = generate_answer(question, candidates, llm=llm)

    sources = []
    for c in candidates[:CONTEXT_TOP_K]:
        p = c["payload"]
        sources.append({
            "title": p.get("title") or p.get("file_name", ""),
            "author": p.get("author", ""),
            "year": p.get("year", ""),
            "page": p.get("page_number", ""),
            "source_file_id": p.get("source_file_id", ""),
            "rrf_score": round(c.get("rrf_score", 0.0), 4),
            "rerank_score": round(c.get("rerank_score", 0.0), 4) if "rerank_score" in c else None,
        })

    return {"answer": answer, "sources": sources, "candidates": candidates}


# ── HTTP GUI server ───────────────────────────────────────────────────────────

_PIPELINE_QUEUE_PATH = Path(__file__).parent / "pipeline_queue.json"

def _read_pipeline_queue() -> dict:
    if _PIPELINE_QUEUE_PATH.exists():
        try:
            return json.loads(_PIPELINE_QUEUE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"index_queue": [], "graph_queue": []}

def _write_pipeline_queue(data: dict) -> None:
    _PIPELINE_QUEUE_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

_INDEXER_OUT = Path(__file__).parent / "indexer_out.txt"
_INDEXER_ERR = Path(__file__).parent / "indexer_err.txt"
_indexer_proc: Optional[subprocess.Popen] = None

def _find_external_indexer_pid() -> Optional[int]:
    """Detect a build_index.py process started outside this server."""
    try:
        import psutil
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmd = " ".join(p.info["cmdline"] or [])
                if "build_index.py" in cmd:
                    return p.info["pid"]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        pass
    return None

def get_indexer_status() -> dict:
    global _indexer_proc
    running = _indexer_proc is not None and _indexer_proc.poll() is None
    exit_code = None
    pid = None
    if _indexer_proc is not None:
        pid = _indexer_proc.pid
        if not running:
            exit_code = _indexer_proc.returncode
    if not running:
        ext_pid = _find_external_indexer_pid()
        if ext_pid:
            return {"running": True, "pid": ext_pid, "exit_code": None, "external": True}
    return {"running": running, "pid": pid, "exit_code": exit_code, "external": False}

def run_indexer(device: str = "cuda") -> dict:
    global _indexer_proc
    if _indexer_proc is not None and _indexer_proc.poll() is None:
        return {"error": "Indexer already running", "pid": _indexer_proc.pid}
    ext_pid = _find_external_indexer_pid()
    if ext_pid:
        return {"error": "Indexer already running (external)", "pid": ext_pid}
    queue = _read_pipeline_queue()
    args = [sys.executable, "-u", str(Path(__file__).parent / "build_index.py"), "--device", device]
    _INDEXER_OUT.write_text("", encoding="utf-8")
    _INDEXER_ERR.write_text("", encoding="utf-8")
    _indexer_proc = subprocess.Popen(
        args,
        cwd=str(Path(__file__).parent),
        stdout=_INDEXER_OUT.open("w", encoding="utf-8", errors="replace"),
        stderr=_INDEXER_ERR.open("w", encoding="utf-8", errors="replace"),
    )
    return {"started": True, "pid": _indexer_proc.pid, "device": device}

def stop_indexer() -> dict:
    global _indexer_proc
    if _indexer_proc is None or _indexer_proc.poll() is not None:
        _indexer_proc = None
        return {"stopped": True, "note": "no process to stop"}
    pid = _indexer_proc.pid
    _indexer_proc.terminate()
    try:
        _indexer_proc.wait(timeout=6)
    except subprocess.TimeoutExpired:
        _indexer_proc.kill()
    _indexer_proc = None
    return {"stopped": True, "pid": pid}

def get_indexer_log(lines: int = 80) -> dict:
    out, err = "", ""
    if _INDEXER_OUT.exists():
        try:
            text = _INDEXER_OUT.read_text(encoding="utf-8", errors="replace")
            out = "\n".join(text.splitlines()[-lines:])
        except Exception:
            pass
    if _INDEXER_ERR.exists():
        try:
            text = _INDEXER_ERR.read_text(encoding="utf-8", errors="replace")
            err = "\n".join(text.splitlines()[-20:])
        except Exception:
            pass
    return {"stdout": out, "stderr": err}


def get_pipeline_status() -> dict:
    """Return per-file ingestion and graph status from pipeline_state.json + graph_state.json."""
    base = Path(__file__).parent

    pipeline_files: dict = {}
    state_path = base / "pipeline_state.json"
    if state_path.exists():
        try:
            pipeline_files = json.loads(state_path.read_text(encoding="utf-8")).get("files", {})
        except Exception:
            pass

    graph_files: dict = {}
    graph_state_path = base / "graph_state.json"
    if graph_state_path.exists():
        try:
            graph_files = json.loads(graph_state_path.read_text(encoding="utf-8")).get("files", {})
        except Exception:
            pass

    queue_data = _read_pipeline_queue()
    queued_index = {e["path"] for e in queue_data.get("index_queue", []) if isinstance(e, dict)}
    queued_graph  = {e["path"] for e in queue_data.get("graph_queue",  []) if isinstance(e, dict)}

    all_paths = (
        set(pipeline_files.keys()) |
        set(graph_files.keys()) |
        queued_index | queued_graph
    )

    files = []
    total_chunks = indexed_count = graphed_count = 0

    for path in sorted(all_paths):
        if not path:
            continue
        idx  = pipeline_files.get(path, {})
        grph = graph_files.get(path, {})
        chunks = idx.get("chunks_indexed", 0)
        total_chunks += chunks

        idx_status = idx.get("status", "queued" if path in queued_index else "not_started")
        if idx_status == "indexed":
            indexed_count += 1

        grph_status = grph.get("status", "queued" if path in queued_graph else "pending")
        if grph_status == "complete":
            graphed_count += 1

        files.append({
            "path":                   path,
            "file_name":              Path(path).name,
            "folder":                 str(Path(path).parent),
            "index_status":           idx_status,
            "chunks_indexed":         chunks,
            "index_error":            idx.get("error"),
            "index_updated_at":       idx.get("updated_at"),
            "graph_status":           grph_status,
            "objects_extracted":      grph.get("objects_extracted", 0),
            "relationships_extracted": grph.get("relationships_extracted", 0),
            "era_distribution":       grph.get("era_distribution", {}),
            "graph_updated_at":       grph.get("updated_at"),
        })

    return {
        "summary": {
            "total_files":  len(files),
            "indexed":      indexed_count,
            "graphed":      graphed_count,
            "total_chunks": total_chunks,
            "queued_index": len(queue_data.get("index_queue", [])),
            "queued_graph": len(queue_data.get("graph_queue", [])),
        },
        "files": files,
        "queue": queue_data,
    }


def get_index_status() -> dict:
    """Return metadata about the current Qdrant index for the /api/status endpoint."""
    try:
        client = get_qdrant()
        if not client or not collection_exists(client):
            return {"status": "no_index", "chunk_count": 0, "collection": COLLECTION_NAME}
        info = client.get_collection(COLLECTION_NAME)
        count = info.points_count or 0
        return {
            "status":       "ready",
            "chunk_count":  count,
            "collection":   COLLECTION_NAME,
            "dense_model":  DENSE_MODEL,
            "sparse_model": SPARSE_MODEL,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc), "chunk_count": 0, "collection": COLLECTION_NAME}


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class _QueryHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler: serves query_gui.html + /api/query + /api/status."""

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _serve_static(self, fs_path: Path, max_age: int = 0) -> None:
        if not fs_path.exists() or not fs_path.is_file():
            self._json({"error": "not found"}, 404)
            return
        mime, _ = mimetypes.guess_type(str(fs_path))
        body = fs_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        if max_age > 0:
            self.send_header("Cache-Control", f"max-age={max_age}, stale-while-revalidate=60")
        else:
            self.send_header("Cache-Control", "no-cache")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type",   "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            gui = Path(__file__).parent / "query_gui.html"
            if not gui.exists():
                self._json({"error": "query_gui.html not found"}, 404)
                return
            body = gui.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type",   "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/status":
            self._json(get_index_status())
        elif path == "/api/pipeline/status":
            self._json(get_pipeline_status())
        elif path == "/api/pipeline/queue":
            self._json(_read_pipeline_queue())
        elif path == "/api/pipeline/indexer":
            self._json(get_indexer_status())
        elif path == "/api/pipeline/log":
            self._json(get_indexer_log())
        elif path == "/vocabulary":
            self._serve_static(Path(__file__).parent / "Genesis_Tag_Vocabulary.json")
        elif path == "/sacred_timeline_enriched.json":
            enriched = Path(__file__).parent / "sacred_timeline_enriched.json"
            if enriched.exists():
                self._serve_static(enriched, max_age=30)  # re-enriched every few minutes by pipeline
            else:
                self._json({"error": "sacred_timeline_enriched.json not yet generated"}, 404)
        elif path in ("/sacred_timeline_current.json", "/sacred_timeline_5-10-2026.json"):
            _tl_candidates = [
                Path(__file__).parent.parent / "sacred-timeline" / "public" / "sacred_timeline_current.json",
                Path(__file__).parent / "sacred_timeline_5-10-2026.json",
            ]
            for _p in _tl_candidates:
                if _p.exists():
                    self._serve_static(_p, max_age=300)  # static snapshot, cache for 5 min
                    return
            self._json({"error": "timeline data not found"}, 404)
        elif path in ("/constellation", "/constellation/"):
            self._serve_static(Path(__file__).parent / "constellation" / "dist" / "index.html")
        elif path.startswith("/constellation/"):
            rel = path[len("/constellation/"):]
            self._serve_static(Path(__file__).parent / "constellation" / "dist" / rel)
        elif path in ("/scriptorium", "/scriptorium/"):
            self._serve_static(Path(__file__).parent / "scriptorium" / "dist" / "index.html")
        elif path.startswith("/scriptorium/"):
            rel = path[len("/scriptorium/"):]
            self._serve_static(Path(__file__).parent / "scriptorium" / "dist" / rel)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length))
        except Exception:
            self._json({"error": "invalid JSON body"}, 400)
            return

        if path == "/api/pipeline/run":
            device = body.get("device", "cuda")
            self._json(run_indexer(device))
            return
        if path == "/api/pipeline/stop":
            self._json(stop_indexer())
            return
        if path == "/api/pipeline/queue":
            # body: {"paths": [...], "operation": "index"|"graph"|"dequeue", "priority": 0}
            op    = body.get("operation", "index")
            paths = [p for p in (body.get("paths") or []) if isinstance(p, str) and p]
            if not paths and op not in ("clear_index", "clear_graph"):
                self._json({"error": "paths required"}, 400)
                return
            queue = _read_pipeline_queue()
            import datetime
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()

            if op == "clear_index":
                queue["index_queue"] = []
            elif op == "clear_graph":
                queue["graph_queue"] = []
            elif op == "dequeue_index":
                queue["index_queue"] = [e for e in queue.get("index_queue", []) if e.get("path") not in paths]
            elif op == "dequeue_graph":
                queue["graph_queue"] = [e for e in queue.get("graph_queue", []) if e.get("path") not in paths]
            elif op == "graph":
                existing = {e["path"] for e in queue.get("graph_queue", [])}
                for p in paths:
                    if p not in existing:
                        queue.setdefault("graph_queue", []).append({
                            "path": p, "priority": body.get("priority", 0),
                            "queued_at": now, "ontology_pass": 1,
                        })
            else:  # "index"
                existing = {e["path"] for e in queue.get("index_queue", [])}
                for p in paths:
                    if p not in existing:
                        queue.setdefault("index_queue", []).append({
                            "path": p, "priority": body.get("priority", 0),
                            "queued_at": now,
                        })
            _write_pipeline_queue(queue)
            self._json({"ok": True, "queue": queue})
            return

        if path != "/api/query":
            self._json({"error": "not found"}, 404)
            return

        question = (body.get("question") or "").strip()
        llm      = body.get("llm", "openrouter")
        if llm not in ("claude", "openai", "openrouter"):
            llm = "openrouter"
        if not question:
            self._json({"error": "question is required"}, 400)
            return
        t0 = time.time()
        result = query(question, verbose=False, llm=llm)
        result["elapsed_s"] = round(time.time() - t0, 2)
        # Strip non-serialisable 'candidates' key (contains numpy arrays)
        result.pop("candidates", None)
        self._json(result)

    def address_string(self) -> str:
        return self.client_address[0]  # skip reverse-DNS lookup (avoids 3s delay per request)

    def log_message(self, *_) -> None:
        pass  # suppress access log noise


def run_http_server(port: int = 5175, llm: str = "openrouter") -> None:
    """Pre-load models, then serve the GUI on localhost:{port}."""
    console.print("[dim]Pre-loading embedding models…[/dim]", end=" ")
    get_dense_model()
    get_sparse_model()
    if BGE_RERANKER_OK:
        get_reranker()
    console.print("[green]ready[/green]")

    server = _ThreadedHTTPServer(("localhost", port), _QueryHandler)
    console.print(
        f"[bold green]Query GUI →[/bold green] [underline]http://localhost:{port}[/underline]"
        f"  [dim](LLM default: {llm} · Ctrl-C to stop)[/dim]"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[dim]Server stopped.[/dim]")


# ── MCP server ────────────────────────────────────────────────────────────────

def run_mcp_server() -> None:
    if not MCP_OK:
        console.print("[red]MCP not available. Install: pip install mcp[/red]")
        sys.exit(1)

    # Pre-load models before serving
    console.print("[dim]Pre-loading embedding models for MCP server...[/dim]")
    get_dense_model()
    get_sparse_model()

    mcp = FastMCP("als-library-rag")

    @mcp.tool()
    def query_library(question: str) -> str:
        """
        Query the Auguste Laurent Society library using RAG.
        Returns a scholarly answer with citations from the indexed collection.
        """
        result = query(question)
        answer = result["answer"]
        if result["sources"]:
            src_lines = []
            for i, s in enumerate(result["sources"], 1):
                parts = [s["title"]]
                if s["author"]:
                    parts.insert(0, s["author"])
                if s["year"]:
                    parts.append(str(s["year"]))
                src_lines.append(f"[{i}] {', '.join(p for p in parts if p)}")
            answer += "\n\n**Sources retrieved:**\n" + "\n".join(src_lines)
        return answer

    @mcp.tool()
    def find_sources(topic: str, top_k: int = 5) -> str:
        """
        Find the most relevant sources in the library for a given topic.
        Returns titles, authors, and relevance scores without generating prose.
        """
        client = get_qdrant()
        if not client or not collection_exists(client):
            return "No index available. Run build_index.py first."
        dense_vec, sparse_vec = embed_query(topic)
        candidates = retrieve(client, dense_vec, sparse_vec)[:top_k]
        lines = []
        for i, c in enumerate(candidates, 1):
            p = c["payload"]
            title = p.get("title") or p.get("file_name", "unknown")
            author = p.get("author", "")
            year = p.get("year", "")
            score = round(c.get("rrf_score", 0.0), 4)
            lines.append(f"{i}. {title}" + (f" — {author}" if author else "") + (f" ({year})" if year else "") + f" [score: {score}]")
        return "\n".join(lines) if lines else "No results found."

    console.print("[green]MCP server starting...[/green]")
    mcp.run()


# ── Interactive REPL ──────────────────────────────────────────────────────────

def print_sources_table(sources: list[dict]) -> None:
    if not sources:
        return
    t = Table(title="Sources", show_header=True, box=None)
    t.add_column("#", width=3)
    t.add_column("Title", max_width=45)
    t.add_column("Author", max_width=25)
    t.add_column("Yr", width=6)
    t.add_column("RRF", width=7)
    for i, s in enumerate(sources, 1):
        t.add_row(
            str(i),
            (s.get("title") or "")[:44],
            (s.get("author") or "")[:24],
            str(s.get("year") or ""),
            f"{s.get('rrf_score', 0):.4f}",
        )
    console.print(t)


def run_interactive(llm: str = "claude") -> None:
    console.print(Panel(
        "[bold]Auguste Laurent Society Library[/bold]\n"
        "[dim]Type a question to query the library.\n"
        "Switch LLM: [bold]/openrouter[/bold] (ensemble) · [bold]/claude[/bold] · [bold]/openai[/bold]\n"
        "[bold]/verbose <question>[/bold] for pipeline details · [bold]/quit[/bold] to exit[/dim]",
        border_style="blue",
    ))
    console.print(f"[dim]LLM: {llm}[/dim]")

    # Pre-load models
    console.print("[dim]Loading models...[/dim]", end=" ")
    get_dense_model()
    get_sparse_model()
    if BGE_RERANKER_OK:
        get_reranker()
    console.print("[green]ready[/green]")

    while True:
        try:
            raw = console.input("\n[bold blue]>[/bold blue] ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not raw:
            continue
        if raw.lower() in ("/quit", "/exit", "quit", "exit"):
            break
        if raw.lower() in ("/claude", "/anthropic"):
            llm = "claude"
            console.print("[dim]Switched to Claude[/dim]")
            continue
        if raw.lower() == "/openai":
            llm = "openai"
            console.print("[dim]Switched to OpenAI[/dim]")
            continue
        if raw.lower() in ("/openrouter", "/or", "/ensemble"):
            llm = "openrouter"
            console.print(f"[dim]Switched to OpenRouter ensemble ({len(OPENROUTER_HYDE_MODELS)} models)[/dim]")
            continue

        verbose = False
        question = raw
        if raw.startswith("/verbose "):
            verbose = True
            question = raw[9:].strip()

        with console.status(f"[dim]Querying ({llm})...[/dim]"):
            result = query(question, verbose=verbose, llm=llm)

        console.print()
        console.print(Panel(
            Markdown(result["answer"]),
            title=f"[bold]{question[:60]}[/bold] [dim]({llm})[/dim]",
            border_style="green",
        ))
        print_sources_table(result["sources"])

    console.print("[dim]Goodbye.[/dim]")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Query the ALS library RAG engine.")
    ap.add_argument("--mcp",   action="store_true", help="Run as MCP server")
    ap.add_argument("--serve", action="store_true", help="Serve the web GUI at http://localhost:PORT")
    ap.add_argument("--port",  type=int, default=5175, help="Port for --serve (default 5175)")
    ap.add_argument("--ask",   type=str, default=None, help="Single question (non-interactive)")
    ap.add_argument("--verbose", action="store_true", help="Show pipeline steps")
    ap.add_argument("--json",  action="store_true", help="Output JSON (for --ask)")
    ap.add_argument("--llm",   choices=["claude", "openai", "openrouter"], default="openrouter",
                    help="Generation backend: 'openrouter' (default, free multi-model ensemble), "
                         "'claude' (Sonnet), or 'openai' (GPT-4o)")
    args = ap.parse_args()

    if args.serve:
        run_http_server(port=args.port, llm=args.llm)
        return

    if args.mcp:
        run_mcp_server()
        return

    if args.ask:
        get_dense_model()
        get_sparse_model()
        result = query(args.ask, verbose=args.verbose, llm=args.llm)
        if args.json:
            print(json.dumps({"answer": result["answer"], "sources": result["sources"]}, indent=2, ensure_ascii=False))
        else:
            console.print(Panel(Markdown(result["answer"]), title=f"{args.ask[:60]} ({args.llm})", border_style="green"))
            print_sources_table(result["sources"])
        return

    run_interactive(llm=args.llm)


if __name__ == "__main__":
    main()
