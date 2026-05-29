/**
 * Genesis Architecture — schema barrel.
 * Import all graph types from here.
 */

export type {
  JsonPatchOperation,
  ChangedBy,
  ChangeType,
  VersioningRecord,
} from './VersioningRecord.js';

export type {
  ObjectType,
  EraId,
  DatePrecision,
  DateRange,
  PublicationType,
  SourceMetadata,
  PassageMetadata,
  SemanticProfile,
  GraphProfile,
  EmbeddingRef,
  GeneratedSummary,
  SummaryScope,
  RefinementComment,
  ReviewStatus,
  ObjectNode,
} from './Object.js';

export type {
  EvidenceKind,
  EvidenceStatus,
  Evidence,
  EdgeSemanticProfile,
  EdgeSummary,
  HomOmegaEdge,
} from './HomOmega.js';

export type {
  AdditionType,
  SuggestionStatus,
  SearchScope,
  ResearchTask,
  EvidenceLink,
  ProposedContent,
  SuggestedAddition,
} from './SuggestedAddition.js';

// ── Top-level snapshot ────────────────────────────────────────────────────────

import type { ObjectNode } from './Object.js';
import type { HomOmegaEdge } from './HomOmega.js';
import type { SuggestedAddition } from './SuggestedAddition.js';
import type { ReviewStatus } from './Object.js';

export interface IngestionFileStatus {
  source_file_id: string;
  file_path: string;
  ocr_status: 'pending' | 'in_progress' | 'complete' | 'failed';
  chunk_count: number;
  embedding_status: 'pending' | 'in_progress' | 'complete' | 'failed';
  edge_count: number;
  review_status: ReviewStatus;
}

export interface PipelineStateRecord {
  current_pass_id: string;
  total_objects: number;
  total_edges: number;
  total_suggestions_open: number;
  ingestion_progress: Record<string, IngestionFileStatus>; // keyed by source_file_id
  last_updated: string; // ISO 8601
}

/**
 * GraphSnapshot — the root document written to
 * sacred-timeline/public/generated/historical-entity-graph.snapshot.json
 * and polled by unified-runtime.ts every 20 seconds.
 */
export type {
  RegionType,
  LayerType,
  BBox,
  HistoDocRegion,
  HistoDocLayer,
  HistoDocDocument,
  HistoDocStateEntry,
  HistoDocState,
} from './HistoDoc.js';

export interface GraphSnapshot {
  schema_version: string;              // semver, e.g., "1.0.0"
  snapshot_id: string;                 // UUID of this snapshot
  generated_at: string;                // ISO 8601
  pass_id: string;                     // pipeline pass that produced this snapshot

  objects: Record<string, ObjectNode>;
  edges: Record<string, HomOmegaEdge>;
  suggested_additions: Record<string, SuggestedAddition>;

  pipeline_state: PipelineStateRecord;
}
