#!/usr/bin/env python3
"""
GENESIS ARCHITECTURE — Gemini Pipeline
Five workers, Rich terminal dashboard, quota management, resumable.
Run: python gemini_pipeline.py
Keys: [1-5] toggle worker  [P] pause  [Q] quit  [N] run normalize
"""

from __future__ import annotations

import datetime
import json
import msvcrt
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Rich imports
# ---------------------------------------------------------------------------
try:
    from rich.columns import Columns
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("ERROR: rich not installed. Run: pip install rich")
    sys.exit(1)

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("ERROR: google-genai not installed. Run: pip install google-genai")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
SACRED_TIMELINE_DIR = BASE_DIR.parent / "sacred-timeline"


def _find(*candidates) -> Path | None:
    for p in candidates:
        path = Path(p)
        if path.exists():
            return path
    return None


TIMELINE_JSON = _find(
    SACRED_TIMELINE_DIR / "public" / "sacred_timeline_current.json",
    SACRED_TIMELINE_DIR / "sacred_timeline_current.json",
    BASE_DIR / "sacred_timeline_5-10-2026.json",
)
TAG_VOCAB_PATH = _find(
    SACRED_TIMELINE_DIR / "Genesis_Tag_Vocabulary.txt",
    BASE_DIR / "Genesis_Tag_Vocabulary.txt",
)
CALCULI_EXPORT_PATH = _find(
    SACRED_TIMELINE_DIR / "Calculi_0-2_Export_5-10-2026.txt",
    BASE_DIR / "Calculi_0-2_Export_5-10-2026.txt",
)

PIPELINE_STATE_PATH = BASE_DIR / "pipeline_state.json"

# JSONL output files
W1_RETRIEVAL = BASE_DIR / "w1_retrieval_packets.jsonl"
W1_MATH = BASE_DIR / "w1_math_reconstruction.jsonl"
W1_HANDOFF = BASE_DIR / "w1_claude_handoff.jsonl"
W2_INDEX = BASE_DIR / "w2_index_records.jsonl"
W2_QUEUE = BASE_DIR / "w2_queue_entities.jsonl"
W2_PROPOSED = BASE_DIR / "w2_proposed_additions.jsonl"
W3_EMBEDDINGS = BASE_DIR / "w3_embeddings.jsonl"
W3_RELS = BASE_DIR / "w3_relationships.jsonl"
W3_LINEAGES = BASE_DIR / "w3_lineages.jsonl"
W4_ENRICHED = BASE_DIR / "w4_entities_enriched.jsonl"
W4_RELS = BASE_DIR / "w4_relationships.jsonl"
W4_TAGS = BASE_DIR / "w4_suggested_tags.jsonl"
W5_BIB = BASE_DIR / "w5_bibliography.jsonl"

# ---------------------------------------------------------------------------
# Rate-limiting constants
# ---------------------------------------------------------------------------
DAILY_LIMIT = 490
RPM_LIMIT = 12
SLEEP_BETWEEN_CALLS = 60 / RPM_LIMIT  # ~5 seconds

# ---------------------------------------------------------------------------
# Calculus period summaries (for W1 prompt)
# ---------------------------------------------------------------------------
CALCULUS_SUMMARIES = {
    0: "Antiquity to ~1600 — Sacred Geometry, Ritual Astronomy, Euclidean Geometry, scholastic logic",
    1: "1600–1750 — Scientific Revolution, Mechanics, Fluxions, Newton-Leibniz Calculus",
    2: "1750–1850 — Analysis and Energy, PDE, Variational Methods, Thermodynamics",
    3: "1850–1900 — Fields and Structures, Riemannian Geometry, Field Theory, Projective Geometry",
    4: "1900–1950 — Abstraction and Foundations, Symmetry, Topology, Hilbert/Banach Spaces",
    5: "1950–present — Cold War Applied Mathematics, Operations Research, Control Systems, Signal Processing",
    6: "1980–present — Semantic Computation, Category Theory, Topos, Distributed Systems, AI",
}

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
DEFAULT_STATE = {
    "date": "",
    "total_calls_today": 0,
    "daily_limit": DAILY_LIMIT,
    "file_search_store": {
        "store_name": None,
        "indexed_files": [],
        "failed_files": [],
    },
    "workers": {
        "w1": {"enabled": True,  "last_calculus": 0, "last_entity_index": 0, "calls_today": 0},
        "w2": {"enabled": True,  "last_file": "",    "calls_today": 0},
        "w3": {"enabled": False, "last_file": "",    "calls_today": 0},
        "w4": {"enabled": True,  "last_person": "",  "calls_today": 0},
        "w5": {"enabled": False, "last_person": "",  "calls_today": 0},
    },
}


def load_state() -> dict:
    if PIPELINE_STATE_PATH.exists():
        with open(PIPELINE_STATE_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        # Deep merge with default to pick up any new keys
        state = json.loads(json.dumps(DEFAULT_STATE))
        _deep_merge(state, saved)
        return state
    return json.loads(json.dumps(DEFAULT_STATE))


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def save_state(state: dict) -> None:
    with open(PIPELINE_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _today() -> str:
    return datetime.date.today().isoformat()


def reset_daily_if_needed(state: dict) -> None:
    today = _today()
    if state.get("date") != today:
        state["date"] = today
        state["total_calls_today"] = 0
        for w in state["workers"].values():
            w["calls_today"] = 0


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------
_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            env_file = BASE_DIR / ".env"
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    if line.startswith("GEMINI_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set. Export it or place in .env")
        _client = genai.Client(api_key=api_key)
    return _client


def call_gemini(
    prompt: str,
    store_name: str | None,
    dashboard_data: dict,
) -> tuple[str, dict]:
    """Call Gemini with optional File Search grounding. Returns (text, usage)."""
    client = get_client()

    tools = []
    if store_name:
        tools.append(
            genai_types.Tool(
                file_search=genai_types.FileSearch(
                    file_search_store_names=[store_name]
                )
            )
        )

    config = genai_types.GenerateContentConfig(
        tools=tools if tools else None
    )

    for attempt in range(5):
        try:
            t0 = time.time()
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=config,
            )
            latency = time.time() - t0
            usage = {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0) or 0,
                "output_tokens": getattr(response.usage_metadata, "candidates_token_count", 0) or 0,
                "total_tokens": getattr(response.usage_metadata, "total_token_count", 0) or 0,
                "latency_s": round(latency, 2),
            }
            dashboard_data["last_call_usage"] = usage
            return response.text or "", usage
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg.upper():
                wait = (2 ** attempt) * 5
                dashboard_data["status"] = f"Rate limited — sleeping {wait}s (attempt {attempt+1}/5)"
                time.sleep(wait)
            else:
                if attempt == 4:
                    raise
                time.sleep(3)

    raise RuntimeError("Max retries exceeded")


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------
def parse_json_response(text: str):
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip())
    return json.loads(text)


def parse_jsonl_response(text: str) -> list[dict]:
    """Parse a response that may be JSONL (one object per line) or a JSON array."""
    text = text.strip()
    # Try as JSON array first
    try:
        result = parse_json_response(text)
        if isinstance(result, list):
            return [r for r in result if isinstance(r, dict)]
    except (json.JSONDecodeError, ValueError):
        pass
    # Try as JSONL
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
        except json.JSONDecodeError:
            pass
    return records


def _append_jsonl(path: Path, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _size_str(path: Path) -> str:
    if not path.exists():
        return "—"
    sz = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if sz < 1024:
            return f"{sz:.1f} {unit}"
        sz //= 1024
    return f"{sz:.1f} GB"


def _compact_names(names: list[str], limit: int = 2000) -> str:
    if len(names) <= limit:
        return json.dumps(names)
    return json.dumps(names[:limit]) + f"\n// ... and {len(names)-limit} more"


# ---------------------------------------------------------------------------
# W1 PROMPT
# ---------------------------------------------------------------------------
def build_w1_prompt(calculus_number: int, calculus_name: str, calculus_summary: str,
                    entity_batch: list, vocab_text: str) -> str:
    return f"""You are operating as the Sacred Timeline Gemini Semantic Retrieval and Mathematical Reconstruction Worker for Calculus {calculus_number} ({calculus_name}).

You are NOT the final author. You are a deterministic semantic mining and reconstruction engine. Your outputs will be consumed by Claude for final scholarly synthesis. Optimize for: determinism, structure, semantic completeness, traceability, source grounding, JSON validity, downstream interoperability.

Your File Search store contains 5000+ primary and secondary sources in mathematics, science, and philosophy. Use it. Search for sources relevant to each entity below. When you locate a source, report it accurately with its exact path. When inferring from training knowledge alone, set found_in_drive to false.

——————————————————————————————
GENESIS TAG VOCABULARY — CLOSED
——————————————————————————————
{vocab_text}
——————————————————————————————

CALCULUS PERIOD: {calculus_number} — {calculus_name}
PERIOD SUMMARY: {calculus_summary}

ENTITIES IN THIS PERIOD:
{json.dumps(entity_batch, indent=2, ensure_ascii=False)}

——————————————————————————————
PRIMARY OBJECTIVES
——————————————————————————————
For each entity and the period as a whole:
1. Search the File Search store for relevant books, papers, manuscripts, treatises, lecture notes, translations, commentaries, historical textbooks
2. Extract citations, passages, and mathematical derivations
3. Reconstruct OCR-damaged mathematics conservatively — mark confidence explicitly
4. Normalize historical notation to modern form, preserving the original
5. Generate valid LaTeX for all mathematical content
6. Extract pedagogically useful exercises, mark difficulty level
7. Identify missing figures and sources
8. Produce Claude-ready synthesis packets

——————————————————————————————
ANTI-ANACHRONISM RULE
——————————————————————————————
Prioritize sources historically relevant to Calculus {calculus_number}.
Do NOT retrieve modern textbooks unless needed for notation normalization.
Period-appropriate sources take absolute priority.

——————————————————————————————
MATHEMATICAL RECONSTRUCTION POLICY
——————————————————————————————
When historical or OCR-damaged notation is encountered:
- Preserve original historical form exactly
- Generate normalized modern form
- Generate valid LaTeX
- Mark OCR uncertainties with confidence scores: high / medium / low
- Never hallucinate missing mathematics
- Preserve historically meaningful ambiguities

——————————————————————————————
OUTPUT CONTRACT
——————————————————————————————
Return ONLY valid JSON. No prose. No markdown fences. No commentary.
If output limit approaches, stop cleanly at a complete object boundary
and set continuation_required: true in retrieval_metadata.

{{
  "retrieval_metadata": {{
    "calculus_number": {calculus_number},
    "calculus_name": "{calculus_name}",
    "timestamp": "<ISO timestamp>",
    "entities_processed": [],
    "drive_coverage": "<high|medium|low>",
    "continuation_required": false
  }},
  "semantic_retrieval_packets": [{{
    "packet_id": "<SRP-001-NAME>",
    "ontology_entities": [],
    "associated_calculus_numbers": [],
    "semantic_kernels": [],
    "historical_period_focus": "",
    "retrieved_sources": [{{
      "title": "", "author": "", "date": "", "language": "",
      "source_type": "", "found_in_drive": true, "drive_path": "",
      "historical_relevance_reason": "", "associated_entities": [],
      "associated_kernels": [], "important_pages": [],
      "ocr_quality": "<high|medium|low|failed|null>",
      "math_density": "<high|medium|low|null>",
      "historical_significance": "",
      "pedagogical_usefulness": "<high|medium|low>",
      "confidence": "<high|medium|low>"
    }}],
    "bibliography_expansion_targets": [],
    "historical_priority": "<critical|high|medium|low>",
    "mathematical_priority": "<critical|high|medium|low>",
    "confidence": "<high|medium|low>"
  }}],
  "mathematical_reconstruction_packets": [{{
    "packet_id": "<MRP-001>",
    "source_title": "", "historical_notation": "",
    "normalized_notation": "", "latex_render": "",
    "notation_mapping": [{{"original": "", "semantic": ""}}],
    "historical_interpretation": "", "modern_interpretation": "",
    "derivation_explanation": "", "associated_kernels": [],
    "difficulty_level": "<grade_6_8_introductory|grade_9_10_intermediate|grade_11_12_advanced|undergraduate_bridge|undergraduate_technical>",
    "ocr_uncertainties": [], "confidence": "<high|medium|low>"
  }}],
  "exercise_extraction_packets": [{{
    "exercise_id": "<EXE-001>",
    "source_title": "", "exercise_text_original": "",
    "exercise_text_normalized": "", "latex_version": "",
    "difficulty": "<grade_6_8_introductory|grade_9_10_intermediate|grade_11_12_advanced|undergraduate_bridge|undergraduate_technical>",
    "historical_period": "", "associated_calculus_numbers": [],
    "associated_kernels": [], "pedagogical_use": "",
    "prerequisite_knowledge": "", "confidence": "<high|medium|low>"
  }}],
  "missing_source_queue": [
    {{"item": "", "reason": "", "priority": "<high|medium|low>"}}
  ],
  "ambiguities_and_uncertainties": [
    {{"type": "", "source": "", "detail": "", "mitigation": ""}}
  ],
  "claude_handoff_packets": [{{
    "handoff_id": "<CHP-001>",
    "theme": "", "chapter_candidate": "",
    "ontology_entities": [], "primary_sources": [],
    "secondary_sources": [], "mathematical_targets": [],
    "historical_narrative_threads": [], "pedagogical_threads": [],
    "latex_targets": [], "open_questions": [],
    "confidence": "<high|medium|low>"
  }}]
}}

RULES:
- found_in_drive: false if inferring from training knowledge — never fabricate paths
- Preserve all ambiguities explicitly — never silently suppress uncertainty
- Schools and tags: Genesis Tag Vocabulary for canonical fields; suggested_* for novel terms
- Do NOT ask confirmation questions — output and stop
- Do NOT output progress bars"""


# ---------------------------------------------------------------------------
# W2 PROMPT
# ---------------------------------------------------------------------------
def build_w2_prompt(file_batch: list, timeline_names_compact: str,
                    queue_names_compact: str, vocab_text: str) -> str:
    return f"""You are operating as a deterministic semantic indexing worker for the Genesis Architecture / Sacred Timeline system.

You are NOT an essayist. You are an extraction, classification, validation, deduplication, and structured output worker. Minimize prose. Output budget is precious.

Your File Search store contains the complete Auguste Laurent Society library. Read each file below from the store and index it.

——————————————————————————————
GENESIS TAG VOCABULARY — CLOSED
——————————————————————————————
{vocab_text}
——————————————————————————————

FILES TO INDEX (read each from the File Search store):
{json.dumps(file_batch, indent=2, ensure_ascii=False)}

ALREADY IN TIMELINE (skip these names — do not re-queue):
{timeline_names_compact}

ALREADY IN QUEUE (skip these names — do not re-queue):
{queue_names_compact}

——————————————————————————————
CALCULI ASSIGNMENT RULES
——————————————————————————————
Assign calculi_id by dominant conceptual regime of file content, not file date:
0 = Zeroth: Antiquity – ~1600, Sacred Geometry, Ritual Astronomy, Euclidean Geometry, scholastic logic
1 = First:  1600–1750, Scientific Revolution, Mechanics, Fluxions, Newton-Leibniz Calculus
2 = Second: 1750–1850, Analysis & Energy, PDE, Variational Methods, Thermodynamics
3 = Third:  1850–1900, Fields & Structures, Riemannian Geometry, Field Theory, Projective Geometry
4 = Fourth: 1900–1950, Abstraction & Foundations, Symmetry, Topology, Hilbert/Banach Spaces
5 = Fifth:  1950–present, Cold War Applied Mathematics, Operations Research, Control Systems, Signal Processing
6 = Sixth:  1980–present, Semantic Computation, Category Theory, Topos, Distributed Systems, AI

Category theory, topos theory, sheaves, stacks, categorical logic → default Sixth
unless explicitly situated in an earlier institutional context.
If multiple calculi apply: choose dominant, add Era-Uncertain to pipeline_flags.

——————————————————————————————
SCHOOLS TAGGING RULE
——————————————————————————————
Assign 2–6 schools per file when justified. Never default to one school.
For category-theoretic texts always evaluate:
Grothendieck School, EGA/SGA Lineage, Paris Seminar Tradition,
Topos Theory Tradition, Sheaf-Theoretic Tradition, Category-Theoretic Semantics,
Bourbaki, Structuralism, Lawvere School.
Unknown schools → suggested_schools only.

——————————————————————————————
TAG VALIDATION RULE
——————————————————————————————
Every canonical tag must be an exact match from the Genesis Tag Vocabulary.
Unknown tags → corresponding suggested_* field only. Never paraphrase vocab terms.

——————————————————————————————
OUTPUT CONTRACT
——————————————————————————————
Return ONLY JSONL — one JSON object per line, no wrapper array, no prose.
Four record types, interleaved as processed:

INDEX RECORD (one per file):
{{"record_type":"index","file_name":"","drive_path":"","calculi_id":null,"cognitive_tags":[],"suggested_cognitive_tags":[],"math_tags":[],"suggested_math_tags":[],"scientific_domain_tags":[],"suggested_domain_tags":[],"schools":[],"suggested_schools":[],"pipeline_flags":[],"suggested_pipeline_flags":[],"evidence_notes":"","confidence":"<high|medium|low>"}}

QUEUE ENTITY (one per newly discovered historically significant entity — omit if already in timeline or queue):
{{"record_type":"queue_entity","name":"","entity_type":"<Person|Text|Concept|Event|Place|Institution|Tradition>","domains":["<Science|Religion|Transmission>"],"birth_year":null,"death_year":null,"lifespan_raw":"","country":null,"field":null,"calculus_number":null,"calculus_name":"","assertion_layer":"ai_hypothesis","schools":[],"suggested_schools":[],"scientific_domains":[],"cognitive_tags":[],"math_tags":[],"summary":"","source_files":[],"confidence":"<high|medium|low>","needs_review":false}}

PROPOSED ADDITION (novel tag not in vocabulary):
{{"record_type":"proposed_addition","tag":"","category":"<cognitive|math|domain|school|pipeline>","reason":"","example_file":""}}

PROCESSING REPORT (one per batch, final record):
{{"record_type":"processing_report","files_processed":0,"files_skipped":[],"entities_discovered":0,"duplicate_entities_skipped":0,"proposed_additions_count":0,"last_file_processed":"","next_file":"","continuation_required":true}}

RULES:
- Read each file from the store deeply enough to justify tagging
- Process in the order given — never reshuffle
- Deduplication: normalize names with .lower().strip() before comparing
- Do NOT output prose summaries, commentaries, or explanations
- Do NOT ask confirmation questions — output and stop
- Do NOT output progress bars"""


# ---------------------------------------------------------------------------
# W3 PROMPT
# ---------------------------------------------------------------------------
def build_w3_prompt(file_name: str, known_entities_compact: str) -> str:
    return f"""You are operating as the semantic topology and embedding engine for the Genesis Architecture / Sacred Timeline system.

You build graph structure. Your primary output is weighted edges, semantic neighbor networks, conceptual lineages, and latent cluster assignments.

Your File Search store contains the complete Auguste Laurent Society library.
Read this file from the store: {file_name}

KNOWN ENTITIES (edges may reference these names or suggest new ones):
{known_entities_compact}

——————————————————————————————
OUTPUT CONTRACT
——————————————————————————————
Return ONLY JSONL. One JSON object per line. No prose. No arrays. No markdown.

EMBEDDING RECORD:
{{"record_type":"embedding","entity_name":"","entity_type":"<Person|Institution|School|Concept|Book|Conference|Discipline>","cluster_id":"","semantic_neighbors":[],"centrality_score":0.0,"source_file":"{file_name}","calculi_id":null,"embedding_confidence":"<Explicit|Strong Inference|Weak Inference|Speculative>","historical_regime":"","institutional_context":[],"conceptual_domains":[]}}

RELATIONSHIP RECORD:
{{"record_type":"relationship","source":"","target":"","relation_type":"<influenced|collaborated_with|institutional_peer|reacted_against|philosophical_descendant_of|mathematical_generalization_of|operationalized_by|precursor_to|formalized_by|opposed_to|synthesis_of|historically_adjacent_to|advisor_of|member_of|participant_in|funded_by|institutionalized_by>","weight":0.0,"confidence":"<Explicit|Strong Inference|Weak Inference|Speculative>","evidence_excerpt":"","source_file":"{file_name}","assertion_layer":"ai_hypothesis"}}

LINEAGE RECORD:
{{"record_type":"lineage","lineage_id":"","nodes":[],"description":"","confidence":"<Explicit|Strong Inference|Weak Inference|Speculative>","source_file":"{file_name}"}}

PROCESSING REPORT (final record):
{{"record_type":"processing_report","file":"{file_name}","embeddings_generated":0,"relationships_generated":0,"lineages_generated":0,"latent_clusters_discovered":[],"next_file":"","continuation_required":false}}

RULES:
- Embeddings: combine co-occurrence, conceptual overlap, institutional overlap, bibliographic proximity, shared schools, shared calculi, historical influence
- Never generate random neighbor associations — only historically/mathematically meaningful adjacency
- Speculative inference allowed IF marked confidence = "Speculative"
- If relationship target not in known entities: emit it anyway — normalizer handles it
- Do NOT ask confirmation questions — output and stop
- Do NOT output progress bars"""


# ---------------------------------------------------------------------------
# W4 PROMPT
# ---------------------------------------------------------------------------
def build_w4_prompt(batch: list, all_names_compact: str,
                    vocab_text: str, n_people: int) -> str:
    return f"""You are a deterministic semantic enrichment engine for the Sacred Timeline, a historiographic dataset of {n_people} historical figures spanning mathematics, science, philosophy, and related fields.

Your outputs are consumed by a downstream normalizer. Optimize for JSON validity, schema compliance, and determinism. Do not improvise structure.

——————————————————————————————
GENESIS TAG VOCABULARY — CLOSED
——————————————————————————————
{vocab_text}
——————————————————————————————

PEOPLE TO ENRICH:
{json.dumps(batch, indent=2, ensure_ascii=False)}

VALID RELATION TARGETS (relations must reference only names from this list):
{all_names_compact}

——————————————————————————————
OUTPUT CONTRACT
——————————————————————————————
Return ONLY a valid JSON array. No prose. No markdown fences. No commentary.
One object per person, same order as input.

[
  {{
    "name": "<exactly as given in input>",
    "relations": [
      {{
        "target": "<exact match from valid targets list>",
        "relation_type": "<influenced|taught|commented_on|translated|debated|founded|discovered|published|corresponded_with|associated_with|member_of|cited_in|published_by|developed_concept|mathematical_generalization_of|philosophical_transformation_of|symbolic_analogue_of|preserved_in_tradition|part_of_series|opposed|located_at|active_during>",
        "weight": 0.0,
        "confidence": "<high|medium|low>",
        "evidence_type": "<Explicit|Strong Inference|Weak Inference|Speculative>",
        "provenance": "<source title or null>",
        "assertion_layer": "ai_hypothesis"
      }}
    ],
    "schools": ["<SCHOOLS vocab — exact match only>"],
    "suggested_schools": ["<school names valid but absent from vocab>"],
    "cognitive_tags": ["<COGNITIVE TAGS — exact match only>"],
    "suggested_cognitive_tags": ["<cognitive concepts absent from vocab>"],
    "math_tags": ["<MATH TAGS — exact match only>"],
    "suggested_math_tags": ["<math concepts absent from vocab>"],
    "domain_tags": ["<SCIENTIFIC DOMAIN TAGS — exact match only>"],
    "suggested_domain_tags": ["<domain concepts absent from vocab>"],
    "suggested_pipeline_flags": ["<free-form metadata e.g. Latin-Source, Math-Dense, OCR-Heavy>"],
    "centrality_score": 0.0,
    "cluster_id": "<short thematic cluster label>"
  }}
]

RULES:
- relations: 3–8 per person maximum. Target must exist in valid targets list.
- canonical tag fields: ONLY exact vocab matches. Case-sensitive.
- suggested_* fields: write freely — never omit a term because it lacks a vocab match
- pipeline_flags: DO NOT generate — absent from output
- bibliography: DO NOT include in this call
- Return [] for any array field with no evidence. Never return null for arrays."""


# ---------------------------------------------------------------------------
# W5 PROMPT
# ---------------------------------------------------------------------------
def build_w5_prompt(person: dict, nearby_names_compact: str) -> str:
    return f"""You are a bibliographic retrieval engine for the Sacred Timeline historical dataset.

Your File Search store contains 5000+ sources in mathematics, science, and philosophy. Search it now for sources relevant to the person below. Report accurately what you find. Do not fabricate sources or paths.

PERSON:
{json.dumps(person, indent=2, ensure_ascii=False)}

PEOPLE FOR TAGGING (names within ±150 years — exact matches only):
{nearby_names_compact}

——————————————————————————————
OUTPUT CONTRACT
——————————————————————————————
Return ONLY a valid JSON array. No prose. No markdown fences.
Return [] if no relevant sources found in the store.

[
  {{
    "title": "",
    "author": "",
    "date": "",
    "language": "",
    "source_type": "<book|paper|manuscript|lecture_notes|translation|commentary|treatise|other>",
    "found_in_drive": true,
    "drive_path": "<exact path as it appears in the store, or null>",
    "pages_accessed": 0,
    "text_extractable": true,
    "ocr_quality": "<high|medium|low|failed|null>",
    "math_density": "<high|medium|low|null>",
    "passages_retrieved": 0,
    "gemini_evidence_type": "<Explicit|Strong Inference|Weak Inference|Speculative>",
    "confidence": "<high|medium|low>",
    "people_tags": [
      {{"name": "<exact match from nearby list>", "relation": "<primary_subject|mentioned|cited|opposed|influenced_by>"}}
    ],
    "suggested_people_tags": [
      {{"name": "<name in source not in nearby list>", "relation": "", "note": ""}}
    ],
    "assertion_layer": "ai_hypothesis"
  }}
]

RULES:
- found_in_drive: false if drawing on training knowledge alone — never fabricate store paths
- passages_retrieved: actual count of excerpts pulled, not inferred
- people_tags: only exact name matches from the nearby list
- suggested_people_tags: names found in source text not in the list
- gemini_evidence_type: reflects confidence in source ACCESS, not in the relation
- Do NOT ask confirmation questions — output and stop"""


# ---------------------------------------------------------------------------
# Worker step functions
# ---------------------------------------------------------------------------

def _w1_step(state: dict, all_people: list, vocab_text: str,
             calculi_by_number: dict, dashboard_data: dict) -> dict | None:
    """One W1 call: process a batch of entities from a calculus period."""
    w1 = state["workers"]["w1"]
    store_name = state["file_search_store"].get("store_name")

    last_calc = w1.get("last_calculus", 0)
    last_idx = w1.get("last_entity_index", 0)

    # Build ordered list of (calculus_number, entities) pairs
    calc_order = sorted(calculi_by_number.keys())
    # Find current position in sequence
    for calc_num in calc_order:
        if calc_num < last_calc:
            continue
        entities = calculi_by_number[calc_num]
        start = last_idx if calc_num == last_calc else 0
        if start >= len(entities):
            last_calc = calc_num + 1
            last_idx = 0
            continue

        batch = entities[start:start + 15]
        calc_name = (batch[0].get("calculus_name") or str(calc_num))
        calc_summary = CALCULUS_SUMMARIES.get(calc_num, "")

        prompt = build_w1_prompt(calc_num, calc_name, calc_summary, batch, vocab_text)
        text, usage = call_gemini(prompt, store_name, dashboard_data)

        try:
            result = parse_json_response(text)
            if isinstance(result, dict):
                for packet in result.get("semantic_retrieval_packets", []):
                    _append_jsonl(W1_RETRIEVAL, packet)
                for packet in result.get("mathematical_reconstruction_packets", []):
                    _append_jsonl(W1_MATH, packet)
                for packet in result.get("claude_handoff_packets", []):
                    _append_jsonl(W1_HANDOFF, packet)
        except (json.JSONDecodeError, ValueError) as exc:
            dashboard_data["last_error"] = f"W1 JSON error: {exc}"

        new_idx = start + len(batch)
        if new_idx >= len(entities):
            w1["last_calculus"] = calc_num + 1
            w1["last_entity_index"] = 0
        else:
            w1["last_calculus"] = calc_num
            w1["last_entity_index"] = new_idx

        w1["calls_today"] += 1
        return {"worker": "W1", "label": f"Calculus {calc_num} entities {start}–{new_idx-1}",
                "usage": usage}

    return None  # all done


def _w2_step(state: dict, all_files: list, all_timeline_names: list,
             vocab_text: str, dashboard_data: dict) -> dict | None:
    """One W2 call: index 5 files."""
    w2 = state["workers"]["w2"]
    store_name = state["file_search_store"].get("store_name")
    if not store_name:
        dashboard_data["last_error"] = "W2: no store_name in state"
        return None

    last_file = w2.get("last_file", "")
    start_idx = 0
    if last_file:
        try:
            start_idx = all_files.index(last_file) + 1
        except ValueError:
            start_idx = 0
    if start_idx >= len(all_files):
        return None

    batch = all_files[start_idx:start_idx + 5]

    # Read already-queued entity names from w2_queue for dedup
    queued_names = set()
    if W2_QUEUE.exists():
        with open(W2_QUEUE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        if rec.get("record_type") == "queue_entity" and rec.get("name"):
                            queued_names.add(rec["name"].lower().strip())
                    except json.JSONDecodeError:
                        pass

    timeline_compact = _compact_names(all_timeline_names, 1000)
    queue_compact = _compact_names(sorted(queued_names), 1000)

    prompt = build_w2_prompt(batch, timeline_compact, queue_compact, vocab_text)
    text, usage = call_gemini(prompt, store_name, dashboard_data)

    for rec in parse_jsonl_response(text):
        rt = rec.get("record_type")
        if rt == "index":
            _append_jsonl(W2_INDEX, rec)
        elif rt == "queue_entity":
            _append_jsonl(W2_QUEUE, rec)
        elif rt == "proposed_addition":
            _append_jsonl(W2_PROPOSED, rec)

    w2["last_file"] = batch[-1]
    w2["calls_today"] += 1
    return {"worker": "W2", "label": batch[-1], "usage": usage}


def _w3_step(state: dict, all_files: list, all_names: list,
             dashboard_data: dict) -> dict | None:
    """One W3 call: embed one file."""
    w3 = state["workers"]["w3"]
    store_name = state["file_search_store"].get("store_name")
    if not store_name:
        return None

    last_file = w3.get("last_file", "")
    start_idx = 0
    if last_file:
        try:
            start_idx = all_files.index(last_file) + 1
        except ValueError:
            pass
    if start_idx >= len(all_files):
        return None

    file_name = all_files[start_idx]
    known_compact = _compact_names(all_names, 1500)

    prompt = build_w3_prompt(file_name, known_compact)
    text, usage = call_gemini(prompt, store_name, dashboard_data)

    for rec in parse_jsonl_response(text):
        rt = rec.get("record_type")
        if rt == "embedding":
            _append_jsonl(W3_EMBEDDINGS, rec)
        elif rt == "relationship":
            rec.setdefault("assertion_layer", "ai_hypothesis")
            _append_jsonl(W3_RELS, rec)
        elif rt == "lineage":
            _append_jsonl(W3_LINEAGES, rec)

    w3["last_file"] = file_name
    w3["calls_today"] += 1
    return {"worker": "W3", "label": file_name, "usage": usage}


def _w4_step(state: dict, all_people: list, vocab_text: str,
             all_names: list, dashboard_data: dict) -> dict | None:
    """One W4 call: enrich a batch of 4 people (no Drive grounding)."""
    w4 = state["workers"]["w4"]
    last_person = w4.get("last_person", "")
    start_idx = 0
    if last_person:
        names = [p["name"] for p in all_people]
        try:
            start_idx = names.index(last_person) + 1
        except ValueError:
            pass
    if start_idx >= len(all_people):
        return None

    batch_records = all_people[start_idx:start_idx + 4]
    batch = [
        {
            "name": p["name"],
            "birth_year": p.get("birth_year"),
            "death_year": p.get("death_year"),
            "calculus_number": p.get("calculus_number"),
            "calculus_name": p.get("calculus_name"),
            "field": p.get("field"),
            "country": p.get("country"),
        }
        for p in batch_records
    ]

    all_names_compact = _compact_names(all_names)
    prompt = build_w4_prompt(batch, all_names_compact, vocab_text, len(all_people))
    text, usage = call_gemini(prompt, None, dashboard_data)  # no Drive for W4

    try:
        results = parse_json_response(text)
        if not isinstance(results, list):
            raise ValueError("Expected JSON array")
        # Align results with batch by name if count mismatches
        if len(results) != len(batch):
            by_name = {r.get("name", ""): r for r in results}
            results = [by_name.get(p["name"], {"name": p["name"], "relations": []})
                       for p in batch]

        suggested_fields = ["suggested_schools", "suggested_cognitive_tags",
                            "suggested_math_tags", "suggested_domain_tags",
                            "suggested_pipeline_flags"]

        for result in results:
            # Ensure assertion_layer on each relation
            for rel in result.get("relations", []):
                rel.setdefault("assertion_layer", "ai_hypothesis")
            _append_jsonl(W4_ENRICHED, result)

            for rel in result.get("relations", []):
                edge = {
                    "source": result.get("name"),
                    "target": rel.get("target"),
                    "relation_type": rel.get("relation_type"),
                    "weight": rel.get("weight", 0.5),
                    "confidence": rel.get("confidence", "medium"),
                    "evidence_type": rel.get("evidence_type"),
                    "provenance": rel.get("provenance"),
                    "assertion_layer": "ai_hypothesis",
                }
                _append_jsonl(W4_RELS, edge)

            if any(result.get(f) for f in suggested_fields):
                tag_rec = {"name": result.get("name")}
                for f in suggested_fields:
                    tag_rec[f] = result.get(f, [])
                _append_jsonl(W4_TAGS, tag_rec)

    except (json.JSONDecodeError, ValueError) as exc:
        dashboard_data["last_error"] = f"W4 JSON error: {exc}"

    w4["last_person"] = batch_records[-1]["name"]
    w4["calls_today"] += 1
    return {"worker": "W4", "label": batch_records[-1]["name"], "usage": usage}


def _w5_step(state: dict, all_people: list, dashboard_data: dict) -> dict | None:
    """One W5 call: bibliography for one person."""
    w5 = state["workers"]["w5"]
    store_name = state["file_search_store"].get("store_name")
    if not store_name:
        return None

    last_person = w5.get("last_person", "")
    start_idx = 0
    if last_person:
        names = [p["name"] for p in all_people]
        try:
            start_idx = names.index(last_person) + 1
        except ValueError:
            pass
    if start_idx >= len(all_people):
        return None

    person = all_people[start_idx]
    birth = person.get("birth_year")

    # Nearby names: ±150 years
    nearby = [
        p["name"] for p in all_people
        if p.get("birth_year") is not None
        and birth is not None
        and abs((p.get("birth_year") or 0) - birth) <= 150
    ]

    nearby_compact = _compact_names(nearby, 500)
    prompt = build_w5_prompt(person, nearby_compact)
    text, usage = call_gemini(prompt, store_name, dashboard_data)

    try:
        results = parse_json_response(text)
        if isinstance(results, list):
            for bib in results:
                bib.setdefault("source_person", person["name"])
                bib.setdefault("assertion_layer", "ai_hypothesis")
                _append_jsonl(W5_BIB, bib)
    except (json.JSONDecodeError, ValueError) as exc:
        dashboard_data["last_error"] = f"W5 JSON error: {exc}"

    w5["last_person"] = person["name"]
    w5["calls_today"] += 1
    return {"worker": "W5", "label": person["name"], "usage": usage}


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def _worker_progress(w_key: str, state: dict, all_people: list,
                     all_files: list, calculi_by_number: dict) -> tuple[int, int, str]:
    """Returns (done, total, last_label) for each worker."""
    w = state["workers"][w_key]
    if w_key == "w4":
        names = [p["name"] for p in all_people]
        last = w.get("last_person", "")
        done = names.index(last) + 1 if last in names else 0
        return done, len(names), last
    if w_key == "w5":
        names = [p["name"] for p in all_people]
        last = w.get("last_person", "")
        done = names.index(last) + 1 if last in names else 0
        return done, len(names), last
    if w_key in ("w2", "w3"):
        last = w.get("last_file", "")
        done = all_files.index(last) + 1 if last in all_files else 0
        return done, len(all_files), last.split("\\")[-1] if last else ""
    if w_key == "w1":
        total = sum(len(v) for v in calculi_by_number.values())
        calc = w.get("last_calculus", 0)
        idx = w.get("last_entity_index", 0)
        done = sum(len(calculi_by_number.get(c, [])) for c in calculi_by_number if c < calc) + idx
        return done, total, f"Calculus {calc}"
    return 0, 1, ""


def build_display(state: dict, dashboard_data: dict, all_people: list,
                  all_files: list, calculi_by_number: dict) -> Panel:
    today_used = state.get("total_calls_today", 0)
    daily_limit = state.get("daily_limit", DAILY_LIMIT)
    pct = today_used / daily_limit if daily_limit else 0

    # Reset time
    now = datetime.datetime.now()
    midnight = datetime.datetime.combine(now.date() + datetime.timedelta(days=1),
                                         datetime.time(0, 0))
    remaining = midnight - now
    h, rem = divmod(int(remaining.total_seconds()), 3600)
    m = rem // 60

    bar_len = 30
    filled = int(bar_len * pct)
    quota_bar = f"[{'#' * filled}{'.' * (bar_len - filled)}] {today_used}/{daily_limit} ({pct*100:.0f}%)  Reset in {h:02d}h {m:02d}m"

    # Workers table
    worker_rows = []
    for wk in ["w1", "w2", "w3", "w4", "w5"]:
        w = state["workers"][wk]
        enabled = "[ON ]" if w.get("enabled") else "[OFF]"
        labels = {
            "w1": "Semantic Retrieval + Math Reconstruction  (Drive)",
            "w2": "Alphabetical Drive Indexer                (Drive)",
            "w3": "Semantic Topology + Embedding             (Drive)",
            "w4": "Entity Enrichment                         (Training)",
            "w5": "Bibliography Retrieval                    (Drive)",
        }
        calls = w.get("calls_today", 0)
        worker_rows.append(f"  {enabled} {wk.upper()}  {labels[wk]}  {calls} calls today")

    # Progress
    progress_rows = []
    for wk in ["w1", "w2", "w4", "w5"]:
        if not state["workers"][wk].get("enabled"):
            continue
        done, total, last = _worker_progress(wk, state, all_people, all_files, calculi_by_number)
        pct2 = done / total if total else 0
        bar2 = f"[{'#' * int(20*pct2)}{'.' * (20 - int(20*pct2))}]"
        label = last[:40] if last else "—"
        progress_rows.append(
            f"  {wk.upper()}  {done}/{total}  {bar2}  {pct2*100:.0f}%  last: {label}"
        )

    # Last call
    last_call = dashboard_data.get("last_call", {})
    usage = dashboard_data.get("last_call_usage", {})
    last_line = "—"
    if last_call:
        w = last_call.get("worker", "")
        lbl = last_call.get("label", "")[:40]
        pt = usage.get("prompt_tokens", 0)
        ot = usage.get("output_tokens", 0)
        lat = usage.get("latency_s", 0)
        last_line = f"  {w} · {lbl} · {pt} prompt tokens · {ot} output tokens · {lat}s"

    err = dashboard_data.get("last_error", "")
    status = dashboard_data.get("status", "Running")

    # Output files
    file_rows = [
        (W1_RETRIEVAL, "w1_retrieval_packets.jsonl"),
        (W2_INDEX, "w2_index_records.jsonl"),
        (W3_EMBEDDINGS, "w3_embeddings.jsonl"),
        (W4_ENRICHED, "w4_entities_enriched.jsonl"),
        (W4_RELS, "w4_relationships.jsonl"),
        (W5_BIB, "w5_bibliography.jsonl"),
    ]
    file_lines = [
        f"  {name:<40}  {_size_str(path):>8}  {_count_jsonl(path):>6} records"
        for path, name in file_rows
        if path.exists()
    ]

    text = "\n".join([
        "GENESIS ARCHITECTURE — GEMINI PIPELINE",
        "=" * 60,
        "",
        f"Daily Quota  {quota_bar}",
        "",
        "WORKERS",
        *worker_rows,
        "",
        "PROGRESS",
        *(progress_rows or ["  (no workers running)"]),
        "",
        "LAST CALL",
        last_line,
        *(["  ERROR: " + err] if err else []),
        "",
        "OUTPUT FILES",
        *(file_lines or ["  (none yet)"]),
        "",
        f"Status: {status}",
        "",
        "[Q] Quit   [P] Pause/Resume   [N] Run Normalize   [1-5] Toggle Worker",
    ])

    return Panel(text, title="[bold cyan]Genesis Architecture[/bold cyan]",
                 border_style="cyan")


# ---------------------------------------------------------------------------
# Keyboard input (Windows non-blocking)
# ---------------------------------------------------------------------------

def check_key() -> str | None:
    if msvcrt.kbhit():
        ch = msvcrt.getch()
        if isinstance(ch, bytes):
            try:
                return ch.decode("utf-8").lower()
            except UnicodeDecodeError:
                pass
    return None


def handle_keys(key: str | None, state: dict, dashboard_data: dict) -> str:
    """Returns 'quit', 'normalize', or '' """
    if key is None:
        return ""
    if key == "q":
        return "quit"
    if key == "n":
        return "normalize"
    if key == "p":
        paused = dashboard_data.get("paused", False)
        dashboard_data["paused"] = not paused
        dashboard_data["status"] = "PAUSED" if not paused else "Running"
    for i, wk in enumerate(["w1", "w2", "w3", "w4", "w5"], 1):
        if key == str(i):
            cur = state["workers"][wk].get("enabled", False)
            state["workers"][wk]["enabled"] = not cur
    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load data
    if not TIMELINE_JSON:
        print("ERROR: sacred_timeline_current.json not found")
        sys.exit(1)
    if not TAG_VOCAB_PATH:
        print("ERROR: Genesis_Tag_Vocabulary.txt not found")
        sys.exit(1)

    print("Loading data...")
    with open(TIMELINE_JSON, encoding="utf-8") as f:
        all_people = json.load(f)
    vocab_text = TAG_VOCAB_PATH.read_text(encoding="utf-8")

    all_names = [p["name"] for p in all_people]

    # Build calculi groupings for W1
    calculi_by_number: dict[int, list] = {}
    if CALCULI_EXPORT_PATH:
        try:
            with open(CALCULI_EXPORT_PATH, encoding="utf-8") as f:
                export_data = json.load(f)
            for bundle in export_data.get("entity_bundles", []):
                c = bundle.get("calculus_number")
                if c is not None:
                    calculi_by_number.setdefault(c, []).append(bundle)
        except Exception as exc:
            print(f"Warning: could not parse Calculi export: {exc}")

    if not calculi_by_number:
        # Fall back to grouping the timeline by calculus_number
        for p in all_people:
            c = p.get("calculus_number")
            if c is not None:
                calculi_by_number.setdefault(c, []).append(p)

    # Files for W2/W3 — use indexed_files list from state if available
    state = load_state()
    reset_daily_if_needed(state)

    all_files: list[str] = state["file_search_store"].get("indexed_files", [])
    if not all_files:
        # If no store yet, fill with filenames from timeline as a placeholder
        all_files = [p["name"] for p in all_people]

    dashboard_data: dict = {
        "last_call": {},
        "last_call_usage": {},
        "last_error": "",
        "status": "Running",
        "paused": False,
    }

    # Determine worker rotation order
    worker_fns = {
        "w1": lambda: _w1_step(state, all_people, vocab_text, calculi_by_number, dashboard_data),
        "w2": lambda: _w2_step(state, all_files, all_names, vocab_text, dashboard_data),
        "w3": lambda: _w3_step(state, all_files, all_names, dashboard_data),
        "w4": lambda: _w4_step(state, all_people, vocab_text, all_names, dashboard_data),
        "w5": lambda: _w5_step(state, all_people, dashboard_data),
    }
    worker_order = ["w1", "w2", "w3", "w4", "w5"]
    current_worker_idx = 0

    console = Console()

    with Live(
        build_display(state, dashboard_data, all_people, all_files, calculi_by_number),
        console=console,
        refresh_per_second=2,
        screen=False,
    ) as live:
        while True:
            # Update display
            live.update(build_display(state, dashboard_data, all_people, all_files, calculi_by_number))

            # Quota check
            if state["total_calls_today"] >= state.get("daily_limit", DAILY_LIMIT):
                dashboard_data["status"] = "DAILY QUOTA EXHAUSTED — restart tomorrow"
                live.update(build_display(state, dashboard_data, all_people, all_files, calculi_by_number))
                save_state(state)
                break

            # Keyboard
            key = check_key()
            action = handle_keys(key, state, dashboard_data)
            if action == "quit":
                dashboard_data["status"] = "Saving and quitting..."
                live.update(build_display(state, dashboard_data, all_people, all_files, calculi_by_number))
                save_state(state)
                break
            if action == "normalize":
                live.stop()
                print("\nRunning normalize.py ...\n")
                subprocess.run([sys.executable, str(BASE_DIR / "normalize.py")])
                live.start()

            if dashboard_data.get("paused"):
                time.sleep(0.5)
                continue

            # Find next enabled worker
            attempts = 0
            result = None
            while attempts < len(worker_order):
                wk = worker_order[current_worker_idx % len(worker_order)]
                current_worker_idx += 1
                attempts += 1
                if not state["workers"][wk].get("enabled"):
                    continue
                try:
                    result = worker_fns[wk]()
                    if result is None:
                        # This worker is complete; disable it
                        state["workers"][wk]["enabled"] = False
                        dashboard_data["status"] = f"{wk.upper()} complete"
                        continue
                    dashboard_data["last_call"] = result
                    state["total_calls_today"] += 1
                    save_state(state)
                    break
                except Exception as exc:
                    dashboard_data["last_error"] = f"{wk.upper()}: {exc}"
                    save_state(state)
                    break

            # Check if all workers are done
            if all(not state["workers"][wk].get("enabled") for wk in worker_order):
                dashboard_data["status"] = "ALL WORKERS COMPLETE"
                live.update(build_display(state, dashboard_data, all_people, all_files, calculi_by_number))
                save_state(state)
                break

            # Sleep between calls, polling keyboard every 100ms
            sleep_total = SLEEP_BETWEEN_CALLS
            slept = 0.0
            while slept < sleep_total:
                time.sleep(0.1)
                slept += 0.1
                key = check_key()
                action = handle_keys(key, state, dashboard_data)
                if action == "quit":
                    save_state(state)
                    return
                if action == "normalize":
                    live.stop()
                    print("\nRunning normalize.py ...\n")
                    subprocess.run([sys.executable, str(BASE_DIR / "normalize.py")])
                    live.start()

    print("\nPipeline stopped. State saved to pipeline_state.json")


if __name__ == "__main__":
    main()
