import { calculusColor } from "./constellation-styles";
import type { GraphNode } from "./types";

interface EntityCardProps {
  person: GraphNode;
  position?: { x: number; y: number };
  onOpenScriptorium?: (person: GraphNode) => void;
}

const year = (value: number | null | undefined) => (typeof value === "number" ? String(value) : "");

const FLAG_STATES: Record<string, "green" | "amber" | "red"> = {
  "Era-Assigned": "green",
  "Auto-Tagged": "green",
  "Tagging-Complete": "green",
  "Ontology-Aligned": "green",
  "Edges-Linked": "green",
  "Graph-Ready": "green",
  "Source-Verified": "green",
  "Text-Extracted": "green",
  "Quality-Verified": "green",
  "Graph-Integrated": "green",
  "Graph-Partial": "amber",
  "Relationship-Incomplete": "amber",
  "Needs-Validation": "amber",
  "Low-Confidence-Tags": "amber",
  "Source-Identified": "amber",
  "OCR-Failed": "red",
  "Quality-Low": "red",
};

const flagColor = (flag: string): string => {
  const state = FLAG_STATES[flag] ?? "amber";
  if (state === "green") return "#4f7f5b";
  if (state === "red") return "#9a554c";
  return "#a5792f";
};

function TagList({
  label,
  canonical = [],
  suggested = [],
}: {
  label: string;
  canonical?: string[];
  suggested?: string[];
}) {
  if (canonical.length === 0 && suggested.length === 0) return null;
  return (
    <div className="entity-card__section">
      <span>{label}</span>
      <div>
        {canonical.map((item) => (
          <strong key={item}>{item}</strong>
        ))}
        {suggested.map((item) => (
          <em key={item}>suggested: {item}</em>
        ))}
      </div>
    </div>
  );
}

function FlagChips({ flags = [] }: { flags?: string[] }) {
  if (flags.length === 0) return null;
  return (
    <div className="entity-card__section">
      <span>Pipeline</span>
      <div className="entity-card__flags">
        {flags.map((flag) => (
          <span key={flag} className="entity-card__flag" style={{ borderColor: flagColor(flag), color: flagColor(flag) }}>
            {flag.replace(/-/g, " ")}
          </span>
        ))}
      </div>
    </div>
  );
}

export function EntityCard({ person, position, onOpenScriptorium }: EntityCardProps) {
  const lifespan = [year(person.birth_year), year(person.death_year)].filter(Boolean).join("-");
  const cardStyle = position
    ? ({
        left: Math.min(window.innerWidth - 360, Math.max(24, position.x + 22)),
        top: Math.min(window.innerHeight - 340, Math.max(24, position.y - 32)),
      } as React.CSSProperties)
    : undefined;

  const allFlags = [...(person.pipeline_flags ?? []), ...(person.suggested_pipeline_flags ?? [])];

  return (
    <aside className="entity-card" style={cardStyle}>
      <div className="entity-card__header">
        <div>
          <h2>{person.name}</h2>
          {lifespan ? <p>{lifespan}</p> : null}
        </div>
        <span
          className="entity-card__badge"
          style={{ borderColor: calculusColor(person.calculus_number), color: calculusColor(person.calculus_number) }}
        >
          {person.calculus_name ?? `Calculus ${person.calculus_number ?? "?"}`}
        </span>
      </div>

      <div className="entity-card__meta">
        {person.field ? <span>{person.field}</span> : null}
        {person.country ? <span>{person.country}</span> : null}
        {person.cluster_id ? <span>{person.cluster_id}</span> : null}
        {typeof person.centrality_score === "number" ? <span>centrality {person.centrality_score.toFixed(2)}</span> : null}
      </div>

      {person.pedagogical_significance ? <p className="entity-card__significance">{person.pedagogical_significance}</p> : null}

      <TagList label="Schools" canonical={person.schools} suggested={person.suggested_schools} />
      <TagList label="Math" canonical={person.math_tags} suggested={person.suggested_math_tags} />
      <TagList label="Cognitive" canonical={person.cognitive_tags} suggested={person.suggested_cognitive_tags} />
      <TagList label="Domain" canonical={person.domain_tags} suggested={person.suggested_domain_tags} />
      <FlagChips flags={allFlags} />

      <button className="entity-card__open" type="button" onClick={() => onOpenScriptorium?.(person)}>
        Open in Scriptorium
      </button>
    </aside>
  );
}
