import type { PersonRecord, TimelineDataState } from "./types";

const isPeopleArray = (value: unknown): value is PersonRecord[] =>
  Array.isArray(value) && value.every((item) => Boolean(item) && typeof item === "object" && "name" in item);

const isEnrichedObject = (value: unknown): value is { people: PersonRecord[] } =>
  Boolean(value) &&
  typeof value === "object" &&
  !Array.isArray(value) &&
  "people" in (value as object) &&
  isPeopleArray((value as { people: unknown }).people);

const fetchJson = async (path: string) => {
  const response = await fetch(path, { cache: "default" });
  if (!response.ok) throw new Error(`${path} returned ${response.status}`);
  return response.json() as Promise<unknown>;
};

export async function loadTimelineData(): Promise<TimelineDataState> {
  try {
    const enriched = await fetchJson("/sacred_timeline_enriched.json");
    // normalize.py writes { people: [...], sheaf_package: {}, ... }
    const people = isEnrichedObject(enriched)
      ? enriched.people
      : isPeopleArray(enriched)
        ? enriched
        : null;
    if (people && people.length > 0) {
      const hasRelationships = people.some((person) => (person.semantic_relationships?.length ?? 0) > 0);
      return { people, source: "enriched", awaitingPipeline: !hasRelationships };
    }
  } catch {
    // Fall through to the flat timeline snapshot.
  }

  const fallbackPaths = [
    "/sacred_timeline_current.json",
    "../sacred-timeline/public/sacred_timeline_current.json",
    "../sacred-timeline/sacred_timeline_current.json",
  ];

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

  return { people: [], source: "fallback", awaitingPipeline: true };
}
