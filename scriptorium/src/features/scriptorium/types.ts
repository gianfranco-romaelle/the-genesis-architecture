export interface PersonTag {
  person_id?: string | null;
  name: string;
  relation?: string | null;
  note?: string | null;
}

export interface BibliographyRecord {
  title?: string | null;
  author?: string | null;
  date?: string | number | null;
  language?: string | null;
  source_type?: string | null;
  found_in_drive?: boolean | null;
  drive_path?: string | null;
  pages_accessed?: number | null;
  text_extractable?: boolean | null;
  ocr_quality?: string | null;
  math_density?: string | null;
  passages_retrieved?: number | null;
  gemini_evidence_type?: string | null;
  confidence?: string | null;
  people_tags?: PersonTag[];
  suggested_people_tags?: PersonTag[];
  pipeline_flags?: string[];
}

export type AssertionLayer = "canonical" | "editorial" | "ai_hypothesis";

export interface SemanticRelationship {
  source?: string | null;
  target?: string | null;
  relation_type?: string | null;
  weight?: number | null;
  confidence?: string | null;
  evidence_type?: string | null;
  provenance?: string | null;
  assertion_layer?: AssertionLayer;
}

export interface PersonRecord {
  name: string;
  birth_year?: number | null;
  death_year?: number | null;
  lifespan_raw?: string | null;
  country?: string | null;
  field?: string | null;
  calculus_number?: number | null;
  calculus_name?: string | null;
  person_id?: string | null;
  math_tags?: string[];
  suggested_math_tags?: string[];
  cognitive_tags?: string[];
  suggested_cognitive_tags?: string[];
  domain_tags?: string[];
  suggested_domain_tags?: string[];
  schools?: string[];
  suggested_schools?: string[];
  pipeline_flags?: string[];
  suggested_pipeline_flags?: string[];
  centrality_score?: number | null;
  cluster_id?: string | null;
  pedagogical_significance?: string | null;
  semantic_relationships?: SemanticRelationship[];
  bibliography?: BibliographyRecord[];
}

export interface TimelineDataState {
  people: PersonRecord[];
  source: "enriched" | "fallback" | "missing";
  awaitingPipeline: boolean;
}
