import { useMemo, useState } from "react";
import { calculusColor } from "./constellation-styles";
import { personId } from "./graph-builder";
import type { BibliographyEntry, PersonRecord, SemanticRelationship } from "./types";

interface ScriptoriumSidebarProps {
  person?: PersonRecord;
  people: PersonRecord[];
  onClose: () => void;
  onFocusPerson: (id: string) => void;
}

const clean = (value?: string | number | null) =>
  value === null || value === undefined || value === "" ? "Unknown" : String(value);

const year = (value: number | null | undefined) => (typeof value === "number" ? String(value) : "?");

const lifespan = (person: PersonRecord) => `${year(person.birth_year)}-${year(person.death_year)}`;

const hasMedia = (person: PersonRecord) =>
  (person.bibliography ?? []).some((source) => source.found_in_drive || source.drive_path);

const eraKey = (person: PersonRecord) =>
  person.calculus_number === null || person.calculus_number === undefined
    ? "unassigned"
    : String(person.calculus_number);

const flagTone = (flag: string) => {
  if (["OCR-Failed", "Text-Not-Extractable", "Quality-Low", "Pipeline-Blocked"].includes(flag)) return "red";
  if (
    [
      "Graph-Partial",
      "Relationship-Incomplete",
      "Needs-Validation",
      "Tagging-In-Progress",
      "Low-Confidence-Tags",
      "Source-Identified",
    ].includes(flag)
  ) {
    return "amber";
  }
  if (
    [
      "Era-Assigned",
      "Auto-Tagged",
      "Tagging-Complete",
      "Edges-Linked",
      "Graph-Ready",
      "Source-Verified",
      "Quality-Verified",
      "Text-Extracted",
      "Core-Text",
      "Graph-Integrated",
    ].includes(flag)
  ) {
    return "green";
  }
  return "grey";
};

function Chip({
  children,
  className = "",
  style,
}: {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <span className={`sidebar-chip ${className}`} style={style}>
      {children}
    </span>
  );
}

function TagBlock({
  label,
  canonical = [],
  suggested = [],
  color,
}: {
  label: string;
  canonical?: string[];
  suggested?: string[];
  color: string;
}) {
  if (canonical.length === 0 && suggested.length === 0) return null;
  return (
    <section className="sidebar-tag-block">
      <h3>{label}</h3>
      {canonical.length > 0 ? (
        <div className="sidebar-chip-row">
          {canonical.map((tag) => (
            <Chip key={tag} className="is-canonical" style={{ "--era-color": color } as React.CSSProperties}>
              {tag}
            </Chip>
          ))}
        </div>
      ) : null}
      {suggested.length > 0 ? (
        <div className="sidebar-suggested-row">
          <span>suggested pending review</span>
          <div className="sidebar-chip-row">
            {suggested.map((tag) => (
              <Chip key={tag} className="is-suggested">
                {tag}
              </Chip>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}

function SourceCard({ source }: { source: BibliographyEntry }) {
  const passages = Math.max(0, Math.min(10, source.passages_retrieved ?? 0));
  const ocr = clean(source.ocr_quality).toLowerCase();
  const ocrTone = ocr === "high" ? "green" : ocr === "medium" ? "amber" : "red";

  return (
    <article className="sidebar-source-card">
      <header>
        <span className={`drive-dot ${source.found_in_drive ? "is-found" : "is-inferred"}`}>
          {source.found_in_drive ? "drive" : "inferred"}
        </span>
        <div>
          <h3>{clean(source.title)}</h3>
          <p>
            {clean(source.author)} · {clean(source.date)} · {clean(source.language)}
          </p>
        </div>
      </header>
      <div className="sidebar-scrape-row">
        <div className="passage-bar">
          <i style={{ width: `${passages * 10}%` }} />
        </div>
        <Chip className={`flag-${ocrTone}`}>OCR {clean(source.ocr_quality)}</Chip>
        <Chip>{clean(source.gemini_evidence_type)}</Chip>
      </div>
      {source.people_tags?.length ? (
        <div className="sidebar-people-tags">
          <span>People tagged</span>
          <div className="sidebar-chip-row">
            {source.people_tags.map((tag) => (
              <Chip key={`${tag.name}-${tag.relation}`}>{tag.name} ({tag.relation})</Chip>
            ))}
          </div>
        </div>
      ) : null}
      {source.suggested_people_tags?.length ? (
        <div className="sidebar-people-tags">
          <span>Suggested people</span>
          <div className="sidebar-chip-row">
            {source.suggested_people_tags.map((tag) => (
              <Chip key={`${tag.name}-${tag.relation}-${tag.note ?? ""}`} className="is-suggested">
                {tag.name} ({tag.relation}{tag.note ? ` - ${tag.note}` : ""})
              </Chip>
            ))}
          </div>
        </div>
      ) : null}
      {source.pipeline_flags?.length ? (
        <div className="sidebar-chip-row">
          {source.pipeline_flags.map((flag) => (
            <Chip key={flag} className={`flag-${flagTone(flag)}`}>
              {flag}
            </Chip>
          ))}
        </div>
      ) : null}
    </article>
  );
}

function GalleryCard({
  person,
  isSelected,
  onFocusPerson,
}: {
  person: PersonRecord;
  isSelected: boolean;
  onFocusPerson: (id: string) => void;
}) {
  const id = personId(person);
  const color = calculusColor(person.calculus_number);
  const sourceCount = person.bibliography?.length ?? 0;
  const relationCount = person.semantic_relationships?.length ?? 0;
  const flags = [...(person.pipeline_flags ?? []), ...(person.suggested_pipeline_flags ?? [])];
  const tags = [
    ...(person.schools ?? []),
    ...(person.math_tags ?? []),
    ...(person.cognitive_tags ?? []),
    ...(person.domain_tags ?? []),
  ];

  return (
    <button
      className={`scriptorium-gallery-card${isSelected ? " is-selected" : ""}`}
      onClick={() => onFocusPerson(id)}
      style={{ "--era-color": color } as React.CSSProperties}
      type="button"
    >
      <div className="scriptorium-gallery-card__beacon" />
      <div className="scriptorium-gallery-card__topline">
        <span>{person.calculus_name ?? `Era ${person.calculus_number ?? "?"}`}</span>
        <span>{hasMedia(person) ? "drive" : "record"}</span>
      </div>
      <strong>{person.name}</strong>
      <p>{lifespan(person)} · {clean(person.field)}</p>
      {person.pedagogical_significance ? <em>{person.pedagogical_significance}</em> : null}
      <div className="scriptorium-gallery-card__meta">
        <span>{relationCount} links</span>
        <span>{sourceCount} sources</span>
        {person.centrality_score != null ? <span>{person.centrality_score.toFixed(2)} centrality</span> : null}
      </div>
      <div className="scriptorium-gallery-card__chips">
        {tags.slice(0, 4).map((tag) => (
          <span key={tag}>{tag}</span>
        ))}
        {flags.slice(0, 2).map((flag) => (
          <span key={flag}>{flag}</span>
        ))}
      </div>
    </button>
  );
}

function RelationshipButton({
  relationship,
  peopleByName,
  onFocusPerson,
}: {
  relationship: SemanticRelationship;
  peopleByName: Map<string, PersonRecord>;
  onFocusPerson: (id: string) => void;
}) {
  const linked = peopleByName.get(clean(relationship.target).toLowerCase());
  return (
    <button
      className={`sidebar-relationship${relationship.assertion_layer === "ai_hypothesis" ? " is-hypothesis" : ""}`}
      disabled={!linked}
      onClick={() => linked && onFocusPerson(personId(linked))}
      type="button"
    >
      <span>→</span>
      <strong>{clean(relationship.target)}</strong>
      <em>{clean(relationship.relation_type).replace(/_/g, " ")}</em>
      <small>{relationship.weight != null ? relationship.weight.toFixed(2) : "n/a"}</small>
    </button>
  );
}

export function ScriptoriumSidebar({ person, people, onClose, onFocusPerson }: ScriptoriumSidebarProps) {
  const [mode, setMode] = useState<"wiki" | "gallery">("wiki");
  const [search, setSearch] = useState("");
  const [eraFilter, setEraFilter] = useState("all");
  const [mediaOnly, setMediaOnly] = useState(false);
  const peopleByName = new Map(people.map((item) => [item.name.toLowerCase(), item]));
  const color = person ? calculusColor(person.calculus_number) : "#888888";
  const selectedId = person ? personId(person) : undefined;
  const eraOptions = useMemo(() => {
    const values = Array.from(new Set(people.map(eraKey))).sort((left, right) => {
      if (left === "unassigned") return 1;
      if (right === "unassigned") return -1;
      return Number(left) - Number(right);
    });
    return ["all", ...values];
  }, [people]);

  const galleryPeople = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return people
      .filter((item) => {
        if (eraFilter !== "all" && eraKey(item) !== eraFilter) return false;
        if (mediaOnly && !hasMedia(item)) return false;
        if (!needle) return true;
        const searchable = [
          item.name,
          item.field,
          item.country,
          item.cluster_id,
          ...(item.schools ?? []),
          ...(item.math_tags ?? []),
          ...(item.cognitive_tags ?? []),
          ...(item.domain_tags ?? []),
        ]
          .filter(Boolean)
          .join(" ")
          .toLowerCase();
        return searchable.includes(needle);
      })
      .sort((left, right) => {
        const mediaRank = Number(hasMedia(right)) - Number(hasMedia(left));
        if (mediaRank !== 0) return mediaRank;
        return (right.centrality_score ?? 0) - (left.centrality_score ?? 0) || left.name.localeCompare(right.name);
      });
  }, [eraFilter, mediaOnly, people, search]);

  return (
    <aside className={`scriptorium-sidebar${person ? " is-open" : ""}`} aria-hidden={!person}>
      {person ? (
        <>
          <header className="scriptorium-sidebar__header" style={{ "--era-color": color } as React.CSSProperties}>
            <div>
              <p className="eyebrow">Scriptorium</p>
              <h2>{person.name}</h2>
              <p>
                {lifespan(person)} · {clean(person.country)} · {clean(person.field)}
              </p>
            </div>
            <button className="scriptorium-sidebar__close" type="button" onClick={onClose}>
              Close
            </button>
          </header>

          <div className="scriptorium-sidebar__tabs">
            <button className={mode === "wiki" ? "is-active" : ""} onClick={() => setMode("wiki")} type="button">
              Wiki
            </button>
            <button className={mode === "gallery" ? "is-active" : ""} onClick={() => setMode("gallery")} type="button">
              Gallery
            </button>
          </div>

          {mode === "wiki" ? (
            <>
              <div className="scriptorium-sidebar__meta">
                <Chip className="is-era" style={{ "--era-color": color } as React.CSSProperties}>
                  {person.calculus_name ?? `Calculus ${person.calculus_number ?? "?"}`}
                </Chip>
                {person.cluster_id ? <Chip>{person.cluster_id}</Chip> : null}
                {typeof person.centrality_score === "number" ? <Chip>centrality {person.centrality_score.toFixed(2)}</Chip> : null}
              </div>

              <TagBlock label="Math Tags" canonical={person.math_tags} suggested={person.suggested_math_tags} color={color} />
              <TagBlock label="Cognitive Tags" canonical={person.cognitive_tags} suggested={person.suggested_cognitive_tags} color={color} />
              <TagBlock label="Domain Tags" canonical={person.domain_tags} suggested={person.suggested_domain_tags} color={color} />
              <TagBlock label="Schools" canonical={person.schools} suggested={person.suggested_schools} color={color} />

              {[...(person.pipeline_flags ?? []), ...(person.suggested_pipeline_flags ?? [])].length ? (
                <section className="scriptorium-sidebar__section">
                  <h3>Pipeline Flags</h3>
                  <div className="sidebar-chip-row">
                    {[...(person.pipeline_flags ?? []), ...(person.suggested_pipeline_flags ?? [])].map((flag) => (
                      <Chip key={flag} className={`flag-${flagTone(flag)}`}>
                        {flag}
                      </Chip>
                    ))}
                  </div>
                </section>
              ) : null}

              {person.pedagogical_significance ? (
                <blockquote className="sidebar-significance" style={{ "--era-color": color } as React.CSSProperties}>
                  {person.pedagogical_significance}
                </blockquote>
              ) : null}

              <section className="scriptorium-sidebar__section">
                <h3>Relationships</h3>
                {person.semantic_relationships?.length ? (
                  <div className="sidebar-relationship-list">
                    {person.semantic_relationships.map((relationship, index) => (
                      <RelationshipButton
                        key={`${relationship.target}-${relationship.relation_type}-${index}`}
                        relationship={relationship}
                        peopleByName={peopleByName}
                        onFocusPerson={onFocusPerson}
                      />
                    ))}
                  </div>
                ) : (
                  <p className="sidebar-placeholder">No relationships mapped yet</p>
                )}
              </section>

              <section className="scriptorium-sidebar__section">
                <h3>Bibliography</h3>
                {person.bibliography?.length ? (
                  person.bibliography.map((source, index) => <SourceCard key={`${source.title}-${index}`} source={source} />)
                ) : (
                  <p className="sidebar-placeholder">No sources indexed yet</p>
                )}
              </section>
            </>
          ) : (
            <section className="scriptorium-gallery">
              <div className="scriptorium-gallery__tools">
                <input
                  aria-label="Search Scriptorium gallery"
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="Search names, schools, tags"
                  value={search}
                />
                <label>
                  <input checked={mediaOnly} onChange={(event) => setMediaOnly(event.target.checked)} type="checkbox" />
                  <span>Drive/source refs only</span>
                </label>
              </div>
              <div className="scriptorium-gallery__eras">
                {eraOptions.map((era) => (
                  <button
                    className={eraFilter === era ? "is-active" : ""}
                    key={era}
                    onClick={() => setEraFilter(era)}
                    type="button"
                  >
                    {era === "all" ? "All" : era === "unassigned" ? "Unassigned" : `Era ${era}`}
                  </button>
                ))}
              </div>
              <div className="scriptorium-gallery__grid">
                {galleryPeople.slice(0, 180).map((item) => (
                  <GalleryCard
                    isSelected={selectedId === personId(item)}
                    key={personId(item)}
                    person={item}
                    onFocusPerson={onFocusPerson}
                  />
                ))}
              </div>
              {galleryPeople.length > 180 ? (
                <p className="sidebar-placeholder">Showing the first 180 matches. Refine search or era to narrow the wall.</p>
              ) : null}
            </section>
          )}
        </>
      ) : null}
    </aside>
  );
}
