/**
 * ObjectNode — node schema for the HomΩ knowledge graph.
 *
 * Covers all object types: Person, SourceFile, Institution,
 * VocabularyTerm, Concept, and Notion.
 *
 * Note on Notion: at MVP, Notion is a synonym for Concept. Post-MVP it will
 * become a dynamic semantic field-object analyzed via a Hodge-Helmholtz-de Rham
 * approach (developmental direction, internal polarity, Shannon entropy relations
 * to neighboring Concepts). Treat Notion === Concept until that spec is written.
 */

import type { ChangedBy, VersioningRecord } from './VersioningRecord.js';

// ── Enumerations ─────────────────────────────────────────────────────────────

export type ObjectType =
  | 'Person'
  | 'SourceFile'
  | 'Institution'
  | 'VocabularyTerm'
  | 'Concept'
  | 'Notion';   // synonym for Concept at MVP

/**
 * era_id IS calculus_number for the Sacred Timeline.
 * Calc0 = pre-antiquity, Calc6 = modern. era_confidence records
 * certainty of the designation on [0, 1].
 */
export type EraId = 'Calc0' | 'Calc1' | 'Calc2' | 'Calc3' | 'Calc4' | 'Calc5' | 'Calc6';

export type DatePrecision = 'exact' | 'approximate' | 'century' | 'unknown';

export type ReviewStatus = 'accepted' | 'rejected' | 'pending';

export type SummaryScope =
  | 'whole_source_file'
  | 'person_profile'
  | 'institution_profile'
  | 'vocabulary_term_profile'
  | 'concept_profile'
  | 'notion_profile';

export type PublicationType =
  | 'monograph'
  | 'article'
  | 'manuscript'
  | 'correspondence'
  | 'edited_volume'
  | 'dissertation'
  | 'pamphlet'
  | 'encyclopedia_entry'
  | 'other';

// ── Sub-interfaces ────────────────────────────────────────────────────────────

export interface DateRange {
  start: string | null;     // ISO 8601 or partial date (e.g., "1743", "1743-08-26")
  end: string | null;
  precision: DatePrecision;
}

/** Populated for SourceFile objects; null on all other types. */
export interface SourceMetadata {
  file_path: string;
  zotero_key: string;           // BetterBibTeX citation key
  title: string;
  author: string | string[];
  year: number | null;
  language: string;             // BCP-47: "en", "fr", "de", "la", etc.
  publication_type: PublicationType;
  page_count: number | null;
  checksum: string;             // SHA-256 of source file bytes
  ingestion_batch_id: string;
}

/**
 * Populated for chunk-level objects and evidence anchoring.
 * char_start / char_end index into the extracted plain-text of the page.
 */
export interface PassageMetadata {
  source_file_id: string;
  page_number: number | null;
  section_title: string | null;
  passage_index: number;        // ordinal position of this chunk within the document
  char_start: number;
  char_end: number;
}

export interface SemanticProfile {
  primary_tags: string[];       // Genesis Tag Vocabulary — high-confidence designations
  secondary_tags: string[];     // supporting or contextual tags
  inferred_tags: string[];      // pipeline-inferred, pending review
  neighbor_refs: string[];      // Object IDs of semantically adjacent nodes
  cluster_refs: string[];       // RAPTOR cluster IDs this Object belongs to
  latent_school_refs: string[]; // LightRAG / GraphRAG community IDs
}

export interface GraphProfile {
  degree_in: number;
  degree_out: number;
  centrality_score: number;
  community_id: string;
  canonical_edge_refs: string[]; // HomΩ edge IDs considered most authoritative for this Object
}

export interface EmbeddingRef {
  qdrant_point_id: string;       // Qdrant UUID — primary retrieval anchor
  embedding_namespace: string;   // Qdrant collection name
  embedding_model: string;       // e.g., "jina-embeddings-v3"
  embedding_updated_at: string;  // ISO 8601
}

export interface GeneratedSummary {
  summary_text: string;
  summary_scope: SummaryScope;
  generated_by: ChangedBy;
  confidence: number;            // [0, 1]
  evidence_ids: string[];
  last_updated: string;          // ISO 8601
}

export interface RefinementComment {
  recommendation:
    | 'strengthen'
    | 'weaken'
    | 'split'
    | 'merge'
    | 'leave_unresolved'
    | 'promote_to_notion'
    | 'queue_research';
  reason: string;
  next_action: string;
}

// ── ObjectNode ────────────────────────────────────────────────────────────────

export interface ObjectNode {
  // ── Identity ───────────────────────────────────────────────────────────────
  id: string;                          // slug: e.g., "lavoisier_antoine_1743"
  object_type: ObjectType;
  label: string;
  aliases: string[];

  // ── Temporal ───────────────────────────────────────────────────────────────
  era_id: EraId;                       // calculus_number alias
  era_confidence: number;              // [0, 1]
  date_range: DateRange;

  // ── Source provenance ──────────────────────────────────────────────────────
  source_metadata: SourceMetadata | null;
  passage_metadata: PassageMetadata | null;

  // ── Semantic topology ──────────────────────────────────────────────────────
  semantic_profile: SemanticProfile;
  graph_profile: GraphProfile;

  // ── Vector store anchor ────────────────────────────────────────────────────
  embedding: EmbeddingRef;

  // ── Content summary ────────────────────────────────────────────────────────
  generated_summary: GeneratedSummary;

  // ── Relations ──────────────────────────────────────────────────────────────
  evidence_refs: string[];             // IDs into the shared Evidence store
  relation_refs: string[];             // HomΩ edge IDs incident to this Object

  // ── Pipeline state ─────────────────────────────────────────────────────────
  pipeline_flags: string[];            // Genesis Tag Vocabulary PIPELINE FLAGS subset
  review_status: ReviewStatus;
  suggested_addition_refs: string[];   // IDs into the SuggestedAddition store

  // ── Refinement ─────────────────────────────────────────────────────────────
  refinement_comment: RefinementComment | null;

  // ── Versioning ─────────────────────────────────────────────────────────────
  versioning_record: VersioningRecord;
}
