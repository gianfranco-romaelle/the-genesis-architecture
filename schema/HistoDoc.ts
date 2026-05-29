/**
 * HistoDoc — three-layer historical document schema.
 *
 * Each physical or logical region of a source document has one
 * canonical_region_id that links up to three HistoDocLayer entries:
 *
 *   diplomatic   raw transcription — original spelling, abbreviations,
 *                punctuation; preserves witness-level readings
 *   normalized   standardised text — expanded abbreviations, corrected
 *                hyphenation, Unicode-normalised; reading text
 *   translation  target-language (English) rendering of the region;
 *                when source language === "eng" this is identical to
 *                normalised but is still recorded so downstream tools
 *                can treat all documents uniformly
 *
 * English-only sources always carry all three layers.
 * Non-English sources carry diplomatic + normalized now;
 * translation is deferred to a later pipeline pass.
 */

// ── Enumerations ─────────────────────────────────────────────────────────────

/**
 * Physical or logical role of a document region.
 * Drives LaTeX rendering choices (heading level, footnote style, etc.)
 */
export type RegionType =
  | 'paragraph'
  | 'heading'            // chapter / section title
  | 'subheading'         // subsection / paragraph mark
  | 'footnote'           // author's footnote at bottom of page
  | 'margin_note'        // annotation in the margin
  | 'list_item'
  | 'caption'            // figure / table caption
  | 'table_cell'
  | 'mathematical_expression'
  | 'running_title'      // header / footer running title
  | 'colophon'           // end-of-text colophon or printer's mark
  | 'epigraph';          // prefatory quotation

export type LayerType = 'diplomatic' | 'normalized' | 'translation';

/** Bounding box in PDF user-space points: [x0, y0, x1, y1]. */
export type BBox = [number, number, number, number];

// ── Core types ────────────────────────────────────────────────────────────────

/**
 * A structural unit of a historical document.
 * canonical_region_id is stable: the same UUID links the region record
 * to all its HistoDocLayer entries regardless of how many layers exist.
 */
export interface HistoDocRegion {
  canonical_region_id: string;   // UUID v4
  page_number: number;           // 1-indexed
  region_type: RegionType;
  sequence_index: number;        // reading order within the document (0-indexed)
  bbox?: BBox;                   // absent for plain-text sources or when unavailable
  block_id?: string;             // source-parser internal block reference (PyMuPDF block #)
  parent_region_id?: string;     // e.g. a list_item's containing paragraph region
}

/**
 * One layer of text content for a canonical region.
 * Multiple HistoDocLayer records can share the same canonical_region_id,
 * one per layer_type that has been produced for that region.
 */
export interface HistoDocLayer {
  canonical_region_id: string;
  layer_type: LayerType;
  text: string;
  language: string;              // ISO 639-3: 'eng', 'lat', 'fra', 'heb', 'ara', …
  script?: string;               // ISO 15924: 'Latn', 'Hebr', 'Arab', 'Grek', …
  confidence?: number;           // [0, 1] — model or OCR confidence
  model_used?: string;           // which model generated this layer
                                 // [Claude Sonnet 4.6 suggestion: "claude-haiku-4-5-20251001"
                                 //  for translation and normalisation passes]
  reviewed?: boolean;            // true once a human has verified this layer
  translation_note?: string;     // brief note on translation choices (translation layer only)
}

/**
 * Root document record.
 * Written to histodoc/<safe_stem>.histodoc.json by histodoc_builder.py.
 */
export interface HistoDocDocument {
  document_id: string;           // UUID v4, stable for this source file
  schema_version: number;        // increment on breaking changes; current = 1
  source_path: string;           // library-relative or absolute path to source PDF
  source_file_id?: string;       // ObjectNode.id of the SourceFile node (if indexed)
  source_language: string;       // ISO 639-3 primary language of the original
  title?: string;
  author?: string | string[];
  year?: number;
  page_count?: number;
  regions: HistoDocRegion[];
  layers: HistoDocLayer[];
  latex_template: 'article' | 'memoir' | 'reledpar';
  created_at: string;            // ISO 8601
  updated_at: string;            // ISO 8601
}

// ── Pipeline state entry ──────────────────────────────────────────────────────

/**
 * One entry inside histodoc_state.json, keyed by source_path.
 * Mirrors the shape of pipeline_state.json / graph_state.json.
 */
export interface HistoDocStateEntry {
  document_id: string;
  source_path: string;
  status:
    | 'pending'
    | 'parsed'          // diplomatic + normalized layers extracted
    | 'translated'      // translation layer added
    | 'latex_built'     // .tex file generated
    | 'failed';
  region_count: number;
  layer_counts: Record<LayerType, number>;
  latex_path?: string;
  error?: string;
  updated_at: string;  // ISO 8601
}

/** Shape of histodoc_state.json on disk. */
export interface HistoDocState {
  files: Record<string, HistoDocStateEntry>;  // keyed by source_path
}
