/**
 * SuggestedAddition — standalone schema for proposed graph extensions.
 *
 * Generated during pipeline passes as a side-effect of enrichment. Lives in its
 * own store (not inline on Object or HomΩ). Referenced by suggested_addition_refs[]
 * on both Object and HomΩ.
 *
 * Git interface: each accepted addition is committed to the pipeline branch,
 * linking the suggestion resolution to the authoritative git history via
 * versioning_record.git_commit_sha.
 *
 * Status lifecycle:
 *   open → accepted → (resolves to new Object / HomΩ edge / tag)
 *   open → rejected
 *   open → deferred  (re-queued for a later pass)
 *   open → merged    (consolidated into an existing suggestion)
 */

import type { ChangedBy, VersioningRecord } from './VersioningRecord.js';
import type { ReviewStatus, RefinementComment } from './Object.js';

// ── Enumerations ─────────────────────────────────────────────────────────────

export type AdditionType =
  | 'new_object'
  | 'new_tag'
  | 'new_edge'
  | 'new_research_task'
  | 'new_cluster'
  | 'new_school'
  | 'new_notion';       // synonym for new_object with object_type "Notion" at MVP

export type SuggestionStatus = 'open' | 'accepted' | 'rejected' | 'deferred' | 'merged';

export type SearchScope = 'local_library' | 'web' | 'rag_pipeline' | 'human_review';

// ── Sub-interfaces ────────────────────────────────────────────────────────────

/**
 * Populated when addition_type === 'new_research_task'.
 * result_summary and completed_at are filled in when the task resolves.
 */
export interface ResearchTask {
  query: string;
  search_scope: SearchScope;
  assigned_to: string | null;          // pass_id, agent name, or human reviewer handle
  due_pass_id: string | null;
  result_summary: string | null;       // populated after task completes
  completed_at: string | null;         // ISO 8601
}

/**
 * Partial draft of the proposed entity.
 * For new_object: partial ObjectNode fields.
 * For new_edge: partial HomOmegaEdge fields.
 * For new_tag: { term, definition, vocabulary_section }.
 * For new_cluster / new_school: { label, member_ids[], description }.
 * Typed as Record<string, unknown> to avoid circular imports; validate at runtime.
 */
export type ProposedContent = Record<string, unknown>;

export interface EvidenceLink {
  evidence_id: string;
  passage_id: string;              // Qdrant point UUID
  quote: string | null;
  confidence_contribution: number; // how much this evidence supports the suggestion [0, 1]
}

// ── SuggestedAddition ─────────────────────────────────────────────────────────

export interface SuggestedAddition {
  // ── Identity ───────────────────────────────────────────────────────────────
  id: string;                          // UUID
  addition_type: AdditionType;
  proposed_id: string;                 // candidate ID for the new object/edge/tag
  label: string;
  reason: string;

  // ── Confidence & evidence ──────────────────────────────────────────────────
  confidence: number;                  // [0, 1]
  evidence_links: EvidenceLink[];      // rich evidence with per-item confidence contribution
  evidence_ids: string[];              // flat list for quick lookup
  priority: 'low' | 'medium' | 'high';

  // ── Origin ─────────────────────────────────────────────────────────────────
  origin_object_id: string | null;     // Object that generated this suggestion
  origin_edge_id: string | null;       // HomΩ edge that generated this suggestion
  origin_pass_id: string;              // pipeline pass
  generated_by: ChangedBy;
  generated_at: string;                // ISO 8601

  // ── Proposed content preview ───────────────────────────────────────────────
  proposed_content: ProposedContent | null;

  // ── Semantic tags ──────────────────────────────────────────────────────────
  semantic_tags: string[];             // Genesis Tag Vocabulary terms relevant to this suggestion

  // ── Resolution ─────────────────────────────────────────────────────────────
  status: SuggestionStatus;
  reviewed_by: ChangedBy | null;
  reviewed_at: string | null;          // ISO 8601
  resolution_note: string | null;
  resolved_object_id: string | null;   // if accepted: Object created or modified
  resolved_edge_id: string | null;     // if accepted: HomΩ edge created or modified
  merged_into_id: string | null;       // if merged: ID of the surviving SuggestedAddition

  // ── Research task ──────────────────────────────────────────────────────────
  research_task: ResearchTask | null;

  // ── Pipeline state ─────────────────────────────────────────────────────────
  pipeline_flags: string[];
  review_status: ReviewStatus;         // follows the global review enum

  // ── Refinement ─────────────────────────────────────────────────────────────
  refinement_comment: RefinementComment | null;

  // ── Versioning ─────────────────────────────────────────────────────────────
  versioning_record: VersioningRecord;
}
