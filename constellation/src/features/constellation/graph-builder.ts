import { calculusColor, relationColor } from "./constellation-styles";
import type { GraphEdge, GraphNode, PersonRecord } from "./types";

const slugify = (value: string) =>
  value
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");

export const personId = (person: PersonRecord) =>
  person.person_id || `${slugify(person.name)}_${person.birth_year ?? "unknown"}`;

export interface BuildGraphOptions {
  showAiHypotheses?: boolean;
}

export function buildGraph(people: PersonRecord[], options: BuildGraphOptions = {}) {
  const { showAiHypotheses = false } = options;
  const idByName = new Map<string, string>();
  people.forEach((person) => idByName.set(person.name, personId(person)));

  const degree = new Map<string, number>();
  const edges: GraphEdge[] = [];
  const edgeKeys = new Set<string>();

  people.forEach((person) => {
    person.semantic_relationships?.forEach((rel, index) => {
      const layer = rel.assertion_layer ?? "canonical";
      if (layer === "ai_hypothesis" && !showAiHypotheses) return;

      const source = idByName.get(rel.source) ?? idByName.get(person.name) ?? personId(person);
      const target = idByName.get(rel.target);
      if (!target || source === target) return;

      const key = `${source}|${target}|${rel.relation_type}|${index}`;
      if (edgeKeys.has(key)) return;
      edgeKeys.add(key);

      degree.set(source, (degree.get(source) ?? 0) + 1);
      degree.set(target, (degree.get(target) ?? 0) + 1);

      edges.push({
        id: `edge_${edgeKeys.size}_${source}_${target}`,
        source,
        target,
        relation_type: rel.relation_type,
        weight: rel.weight ?? 0.5,
        color: layer === "ai_hypothesis" ? "#ffffff18" : relationColor(rel.relation_type),
        assertion_layer: layer,
        isSelectionAdjacent: false,
        isDimmed: false,
      });
    });
  });

  const nodes: GraphNode[] = people.map((person) => {
    const id = personId(person);
    const centrality = person.centrality_score ?? 0.3;
    return {
      ...person,
      id,
      label: person.name,
      size: centrality * 12 + 4,
      color: calculusColor(person.calculus_number),
      degree: degree.get(id) ?? 0,
      isSelected: false,
      isNeighbor: false,
      isDimmed: false,
    };
  });

  return { nodes, edges };
}
