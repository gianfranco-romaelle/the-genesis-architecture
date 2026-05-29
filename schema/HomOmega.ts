/**
 * HomΩ — edge schema for the knowledge graph.
 *
 * HomΩ(a, b) is an Ω-valued morphism: each edge carries a truth value
 * expressed as (weight, confidence) ∈ [0,1]², where weight is relational
 * strength and confidence is epistemic certainty. Together they define a
 * fuzzy/probabilistic edge in the topos-theoretic sense.
 *
 * Evidence kinds use a fine-grained epistemic distinction:
 *   in_source_text_direct     — quote found directly in a library document
 *   secondary_source_inferred — inferred from a secondary library source
 *   llm_training_knowledge    — from LLM training data; no library source exists
 *   llm_prior_pass_recalled   — derived/cached from a previous pipeline pass
 *   research_task_queued      — evidence not yet located; task is queued
 */

import type {
  ChangedBy, VersioningRecord,
} from './VersioningRecord.js';
import type {
  EraId, DateRange, EmbeddingRef, RefinementComment, ReviewStatus,
} from './Object.js';

// ── Evidence ──────────────────────────────────────────────────────────────────

export type EvidenceKind =
  | 'in_source_text_direct'
  | 'secondary_source_inferred'
  | 'llm_training_knowledge'     // LLM knew this from training data; not in library
  | 'llm_prior_pass_recalled'    // derived/cached in a previous pipeline pass
  | 'research_task_queued';      // no source yet; research task created

export type EvidenceStatus = 'verified' | 'needs_review' | 'generated' | 'queued';

export interface Evidence {
  id: string;
  kind: EvidenceKind;
  source_id: string;             // Zotero key or SourceFile Object ID
  /**
   * Qdrant point UUID of the chunk this evidence is anchored to.
   * This is the bridge from the graph to the vector store: given any edge,
   * its evidence passages are directly retrievable as Qdrant points, enabling
   * semantic search over the evidence base.
   */
  passage_id: string;
  quote: string | null;
  page: number | null;
  status: EvidenceStatus;
}

// ── Edge-local sub-interfaces ─────────────────────────────────────────────────

export interface EdgeSemanticProfile {
  primary_tags: string[];        // Genesis Tag Vocabulary
  secondary_tags: string[];
  inferred_tags: string[];
}

export interface EdgeSummary {
  summary_text: string;
  generated_by: ChangedBy;
  confidence: number;            // [0, 1]
  evidence_ids: string[];
  last_updated: string;          // ISO 8601
}

// ── HomΩ edge ─────────────────────────────────────────────────────────────────

export interface HomOmegaEdge {
  // ── Identity ───────────────────────────────────────────────────────────────
  id: string;                          // UUID
  relation_type: string;               // e.g., "influenced_by", "collaborated_with", "contested"
  source_object_id: string;
  target_object_id: string;

  // ── Weights ────────────────────────────────────────────────────────────────
  weight: number;                      // relational strength [0, 1]
  confidence: number;                  // epistemic certainty [0, 1]
  created_by: ChangedBy;

  // ── Temporal scope ─────────────────────────────────────────────────────────
  era_id: EraId;
  era_confidence: number;              // [0, 1]
  date_range: DateRange;               // when this relation held

  // ── Semantic tags ──────────────────────────────────────────────────────────
  semantic_profile: EdgeSemanticProfile;

  // ── Vector store anchor ────────────────────────────────────────────────────
  // Embedding of the relation itself (relation_type + summary), enabling
  // semantic search over edge types and retrieval of structurally similar claims.
  embedding: EmbeddingRef;

  // ── Content summary ────────────────────────────────────────────────────────
  edge_summary: EdgeSummary;

  // ── Evidence ───────────────────────────────────────────────────────────────
  evidence: Evidence[];                // inline evidence items specific to this edge
  evidence_refs: string[];             // pointers into the shared Evidence store (for dedup)

  // ── Pipeline state ─────────────────────────────────────────────────────────
  pipeline_flags: string[];
  review_status: ReviewStatus;
  suggested_addition_refs: string[];   // IDs into the SuggestedAddition store

  // ── Refinement ─────────────────────────────────────────────────────────────
  refinement_comment: RefinementComment | null;

  // ── Versioning ─────────────────────────────────────────────────────────────
  versioning_record: VersioningRecord;
}
