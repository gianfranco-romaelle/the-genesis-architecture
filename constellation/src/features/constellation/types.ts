export interface PeopleTag {
  person_id: string;
  name: string;
  relation: string;
}

export interface SuggestedPeopleTag {
  name: string;
  relation: string;
  note?: string;
}

export interface BibliographyEntry {
  title: string;
  author?: string | null;
  date?: string | null;
  language?: string | null;
  found_in_drive?: boolean | null;
  drive_path?: string | null;
  pages_accessed?: number | null;
  text_extractable?: boolean | null;
  ocr_quality?: string | null;
  math_density?: string | null;
  passages_retrieved?: number | null;
  gemini_evidence_type?: string | null;
  confidence?: string | null;
  people_tags?: PeopleTag[];
  suggested_people_tags?: SuggestedPeopleTag[];
  pipeline_flags?: string[];
}

export type AssertionLayer = "canonical" | "editorial" | "ai_hypothesis";

export interface SemanticRelationship {
  source: string;
  target: string;
  relation_type: string;
  weight?: number;
  confidence?: string;
  evidence_type?: string;
  provenance?: string;
  assertion_layer?: AssertionLayer;
}

export interface PersonRecord {
  name: string;
  birth_year?: number | null;
  death_year?: number | null;
  country?: string | null;
  field?: string | null;
  calculus_number?: number | null;
  calculus_name?: string | null;
  person_id?: string | null;
  semantic_relationships?: SemanticRelationship[];
  schools?: string[];
  suggested_schools?: string[];
  math_tags?: string[];
  suggested_math_tags?: string[];
  cognitive_tags?: string[];
  suggested_cognitive_tags?: string[];
  domain_tags?: string[];
  suggested_domain_tags?: string[];
  pipeline_flags?: string[];
  suggested_pipeline_flags?: string[];
  centrality_score?: number | null;
  cluster_id?: string | null;
  pedagogical_significance?: string | null;
  bibliography?: BibliographyEntry[];
}

export interface GraphNode extends PersonRecord {
  id: string;
  label: string;
  size: number;
  color: string;
  degree: number;
  isSelected: boolean;
  isNeighbor: boolean;
  isDimmed: boolean;
  isSearchMatch?: boolean;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  relation_type: string;
  weight: number;
  color: string;
  assertion_layer?: AssertionLayer;
  isSelectionAdjacent: boolean;
  isDimmed: boolean;
}

export interface TimelineDataState {
  people: PersonRecord[];
  source: "enriched" | "fallback";
  awaitingPipeline: boolean;
}
