import type { PersonRecord, TimelineDataState } from "./types";

const isPeopleArray = (value: unknown): value is PersonRecord[] =>
  Array.isArray(value) && value.every((item) => Boolean(item) && typeof item === "object" && "name" in item);

const fetchJson = async (path: string) => {
  const response = await fetch(path, { cache: "default" });
  if (!response.ok) throw new Error(`${path} returned ${response.status}`);
  return response.json() as Promise<unknown>;
};

const hasEnrichedFields = (person: PersonRecord) =>
  Boolean(
    person.bibliography?.length ||
      person.semantic_relationships?.length ||
      person.pipeline_flags?.length ||
      person.suggested_math_tags?.length ||
      person.suggested_cognitive_tags?.length ||
      person.suggested_domain_tags?.length ||
      person.suggested_schools?.length,
  );

export interface SuggestedTagEntry {
  tag: string;
  tag_class: string;
  entity_name?: string;
  person_id?: string;
  note?: string;
}

export interface SuggestedPersonEntry {
  name: string;
  relation?: string;
  note?: string;
  source_title?: string;
  entity_name?: string;
}

export interface AiRelationshipEntry {
  source: string;
  target: string;
  relation_type: string;
  weight?: number;
  confidence?: string;
  evidence_type?: string;
  provenance?: string;
  assertion_layer: "ai_hypothesis";
}

export interface EnrichedTopLevel {
  suggested_tags_registry?: SuggestedTagEntry[];
  suggested_people_registry?: SuggestedPersonEntry[];
  ai_relationships?: AiRelationshipEntry[];
}

export async function loadTimelineData(): Promise<TimelineDataState & EnrichedTopLevel> {
  try {
    const enriched = await fetchJson("/sacred_timeline_enriched.json");
    if (isPeopleArray(enriched) && enriched.length > 0) {
      return {
        people: enriched,
        source: "enriched",
        awaitingPipeline: !enriched.some(hasEnrichedFields),
      };
    }

    if (enriched && typeof enriched === "object" && !Array.isArray(enriched)) {
      const obj = enriched as Record<string, unknown>;
      const people = isPeopleArray(obj.people) ? obj.people : null;
      if (people && people.length > 0) {
        return {
          people,
          source: "enriched",
          awaitingPipeline: !people.some(hasEnrichedFields),
          suggested_tags_registry: Array.isArray(obj.suggested_tags_registry)
            ? (obj.suggested_tags_registry as SuggestedTagEntry[])
            : [],
          suggested_people_registry: Array.isArray(obj.suggested_people_registry)
            ? (obj.suggested_people_registry as SuggestedPersonEntry[])
            : [],
          ai_relationships: Array.isArray(obj.ai_relationships)
            ? (obj.ai_relationships as AiRelationshipEntry[])
            : [],
        };
      }
    }
  } catch {
    // Fall through to the flat timeline snapshot.
  }

  const fallbackPaths = ["/sacred_timeline_current.json", "../sacred-timeline/sacred_timeline_current.json"];

  for (const path of fallbackPaths) {
    try {
      const fallback = await fetchJson(path);
      if (isPeopleArray(fallback)) {
        return { people: fallback, source: "fallback", awaitingPipeline: true };
      }
    } catch {
      // Try the next candidate.
    }
  }

  return { people: [], source: "missing", awaitingPipeline: true };
}
