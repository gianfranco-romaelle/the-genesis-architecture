import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  loadTimelineData,
  type AiRelationshipEntry,
  type EnrichedTopLevel,
  type SuggestedPersonEntry,
  type SuggestedTagEntry,
} from "./data-loader";
import type { BibliographyRecord, PersonRecord, PersonTag, SemanticRelationship, TimelineDataState } from "./types";

const CALCULUS_COLORS: Record<string, string> = {
  "0": "#c9a96e",
  "1": "#7ec8c8",
  "2": "#a8d8a8",
  "3": "#b8a0d8",
  "4": "#f0a060",
  "5": "#80b8f0",
  "6": "#d4a0c8",
  null: "#888888",
};

const ERA_TABS = [
  { label: "All", value: "all" },
  { label: "Zeroth", value: "0" },
  { label: "First", value: "1" },
  { label: "Second", value: "2" },
  { label: "Third", value: "3" },
  { label: "Fourth", value: "4" },
  { label: "Fifth", value: "5" },
  { label: "Sixth", value: "6" },
  { label: "Unassigned", value: "unassigned" },
] as const;

const FLAG_GROUPS = {
  green: new Set([
    "Era-Assigned", "Auto-Tagged", "Tagging-Complete", "Edges-Linked", "Graph-Ready",
    "Source-Verified", "Quality-Verified", "Text-Extracted", "Core-Text", "Graph-Integrated",
  ]),
  amber: new Set([
    "Graph-Partial", "Relationship-Incomplete", "Needs-Validation",
    "Tagging-In-Progress", "Low-Confidence-Tags",
  ]),
  red: new Set(["OCR-Failed", "Text-Not-Extractable", "Quality-Low", "Pipeline-Blocked"]),
};

const OCR_STARS: Record<string, string> = {
  high: "★★★",
  medium: "★★✦",
  low: "★✦✦",
  failed: "✦✦✦",
};

const evidenceTone: Record<string, string> = {
  Explicit: "blue",
  "Strong Inference": "teal",
  "Weak Inference": "amber",
  Speculative: "grey",
};

const personId = (person: PersonRecord) => person.person_id || slugPerson(person);

const slugPerson = (person: PersonRecord) =>
  `${person.name}_${person.birth_year ?? "unknown"}`
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_|_$/g, "");

const eraKey = (person: PersonRecord) =>
  person.calculus_number === null || person.calculus_number === undefined
    ? "null"
    : String(person.calculus_number);

const eraColor = (person: PersonRecord) => CALCULUS_COLORS[eraKey(person)] ?? CALCULUS_COLORS.null;

const eraLabel = (person: PersonRecord) =>
  person.calculus_name || (person.calculus_number == null ? "Unknown Era" : `Era ${person.calculus_number}`);

const lifespan = (person: PersonRecord) => {
  if (person.lifespan_raw) return person.lifespan_raw;
  const birth = person.birth_year ?? "?";
  const death = person.death_year ?? "?";
  return `${birth}–${death}`;
};

const clean = (value?: string | number | null) =>
  value === null || value === undefined || value === "" ? "Unknown" : String(value);

const listHas = (values?: unknown[]) => Array.isArray(values) && values.length > 0;

function Chip({
  children,
  className = "",
  style,
  title,
}: {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
  title?: string;
}) {
  return (
    <span className={`chip ${className}`} style={style} title={title}>
      {children}
    </span>
  );
}

function FlagChip({ flag, suggested = false }: { flag: string; suggested?: boolean }) {
  let tone = "grey";
  if (!suggested && FLAG_GROUPS.green.has(flag)) tone = "green";
  if (!suggested && FLAG_GROUPS.amber.has(flag)) tone = "amber";
  if (!suggested && FLAG_GROUPS.red.has(flag)) tone = "red";
  return <Chip className={`flag flag--${tone}`}>{suggested ? `suggested: ${flag}` : flag}</Chip>;
}

function TagRows({
  label,
  canonical,
  suggested,
  color,
}: {
  label: string;
  canonical?: string[];
  suggested?: string[];
  color: string;
}) {
  if (!listHas(canonical) && !listHas(suggested)) return null;
  return (
    <section className="tag-class">
      <h3>{label}</h3>
      {listHas(canonical) ? (
        <div className="chip-row">
          {canonical?.map((tag) => (
            <Chip
              className="chip--canonical"
              key={tag}
              style={{ "--era-color": color } as React.CSSProperties}
            >
              {tag}
            </Chip>
          ))}
        </div>
      ) : null}
      {listHas(suggested) ? (
        <div className="suggested-row">
          <span>suggested (pending review)</span>
          <div className="chip-row">
            {suggested?.map((tag) => (
              <Chip className="chip--suggested" key={tag}>
                ◦ {tag}
              </Chip>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}

function PersonList({
  people,
  selectedId,
  onSelect,
}: {
  people: PersonRecord[];
  selectedId?: string;
  onSelect: (person: PersonRecord) => void;
}) {
  const listRef = useRef<HTMLDivElement | null>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const rowHeight = 74;
  const shouldWindow = people.length > 200;
  const visibleCount = 18;
  const start = shouldWindow ? Math.max(0, Math.floor(scrollTop / rowHeight) - 4) : 0;
  const end = shouldWindow ? Math.min(people.length, start + visibleCount + 8) : people.length;
  const visiblePeople = people.slice(start, end);

  // Only update scrollTop when the visible window would actually shift — avoids
  // re-rendering on every pixel of scroll.
  const handleScroll = useCallback(
    (event: React.UIEvent<HTMLDivElement>) => {
      const next = event.currentTarget.scrollTop;
      const nextStart = Math.max(0, Math.floor(next / rowHeight) - 4);
      const curStart = Math.max(0, Math.floor(scrollTop / rowHeight) - 4);
      if (nextStart !== curStart) setScrollTop(next);
    },
    [scrollTop, rowHeight],
  );

  return (
    <div
      className="person-list"
      onScroll={shouldWindow ? handleScroll : undefined}
      ref={listRef}
      role="listbox"
    >
      <div style={shouldWindow ? { height: people.length * rowHeight, position: "relative" } : undefined}>
        <div style={shouldWindow ? { transform: `translateY(${start * rowHeight}px)` } : undefined}>
          {visiblePeople.map((person) => {
            const id = personId(person);
            return (
              <button
                className={`person-row${selectedId === id ? " is-selected" : ""}`}
                key={id}
                onClick={() => onSelect(person)}
                style={{ "--era-color": eraColor(person) } as React.CSSProperties}
                type="button"
              >
                <i />
                <span>
                  <strong>{person.name}</strong>
                  <small>
                    {lifespan(person)} · {eraLabel(person)}
                  </small>
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function PeopleTags({
  label,
  tags,
  suggested = false,
}: {
  label: string;
  tags?: PersonTag[];
  suggested?: boolean;
}) {
  if (!listHas(tags)) return null;
  return (
    <div className="people-tags">
      <span>{label}</span>
      <div className="chip-row">
        {tags?.map((tag) => (
          <Chip
            className={suggested ? "chip--suggested" : "chip--canonical"}
            key={`${tag.name}-${tag.relation}-${tag.note ?? ""}`}
          >
            {suggested ? "◦ " : ""}
            {tag.name} ({clean(tag.relation)}
            {tag.note ? ` — ${tag.note}` : ""})
          </Chip>
        ))}
      </div>
    </div>
  );
}

function BibliographyCard({ source }: { source: BibliographyRecord }) {
  const passages = Math.max(0, Math.min(10, source.passages_retrieved ?? 0));
  const ocrKey = (source.ocr_quality ?? "").toLowerCase();
  const ocrStars = OCR_STARS[ocrKey] ?? "–";
  const ocrTone = ocrKey === "high" ? "green" : ocrKey === "medium" ? "amber" : "red";
  const evidence = clean(source.gemini_evidence_type);

  return (
    <article className="source-card">
      <header className="source-card__header">
        <span className={`drive-dot ${source.found_in_drive ? "is-found" : "is-inferred"}`}>
          {source.found_in_drive ? "drive" : "inferred"}
        </span>
        <div>
          <h3>{clean(source.title)}</h3>
          <p>
            {clean(source.author)} · {clean(source.date)} · {clean(source.language)}
            {source.source_type ? ` · ${source.source_type}` : ""}
          </p>
          {source.drive_path ? <p className="drive-path">{source.drive_path}</p> : null}
        </div>
      </header>

      <div className="scrape-row">
        <span>Scrape quality</span>
        <div className="passage-bar" aria-label={`${passages} passages retrieved`}>
          <i style={{ width: `${passages * 10}%` }} />
        </div>
        <Chip className={`flag flag--${ocrTone}`} title={`OCR ${clean(source.ocr_quality)}`}>
          {ocrStars} OCR
        </Chip>
        <Chip className={`evidence evidence--${evidenceTone[evidence] ?? "grey"}`}>{evidence}</Chip>
        {typeof source.pages_accessed === "number" ? (
          <span className="scrape-meta">{source.pages_accessed}pp</span>
        ) : null}
        <span className="scrape-meta">{passages}/10 passages</span>
      </div>

      <PeopleTags label="People tagged" tags={source.people_tags} />
      <PeopleTags label="Suggested people" tags={source.suggested_people_tags} suggested />

      {listHas(source.pipeline_flags) ? (
        <div className="source-flags">
          {source.pipeline_flags?.map((flag) => (
            <FlagChip flag={flag} key={flag} />
          ))}
        </div>
      ) : null}
    </article>
  );
}

function RelationshipRow({
  rel,
  onNavigate,
  isAiHypothesis,
}: {
  rel: SemanticRelationship;
  onNavigate?: () => void;
  isAiHypothesis?: boolean;
}) {
  const target = clean(rel.target);
  const weight = rel.weight != null ? Number(rel.weight).toFixed(2) : "–";
  const layer = rel.assertion_layer ?? "canonical";

  return (
    <button
      className={`relationship-row${isAiHypothesis ? " is-hypothesis" : ""}`}
      disabled={!onNavigate}
      onClick={onNavigate}
      title={rel.provenance ? `Source: ${rel.provenance}` : undefined}
      type="button"
    >
      <span className="rel-arrow">→</span>
      <span className="rel-target">{target}</span>
      <span className="rel-type">{clean(rel.relation_type).replace(/_/g, " ")}</span>
      <span className="rel-weight">{weight}</span>
      {layer !== "canonical" ? <span className="rel-layer">{layer.replace("_", " ")}</span> : null}
    </button>
  );
}

function DetailView({
  person,
  peopleById,
  peopleByName,
  onSelectId,
}: {
  person?: PersonRecord;
  peopleById: Map<string, PersonRecord>;
  peopleByName: Map<string, PersonRecord>;
  onSelectId: (id: string) => void;
}) {
  if (!person) {
    return (
      <section className="detail-empty">
        <p className="eyebrow">Scriptorium</p>
        <h2>Select a figure</h2>
        <p>Browse the roster on the left to open an archival record.</p>
      </section>
    );
  }

  const color = eraColor(person);
  const bibliography = person.bibliography ?? [];
  const relationships = person.semantic_relationships ?? [];

  return (
    <article className="detail-view" style={{ "--era-color": color } as React.CSSProperties}>
      <header className="detail-header">
        <div>
          <h1>{person.name}</h1>
          <p>
            {lifespan(person)} · {clean(person.country)} · {clean(person.field)}
          </p>
          <span>
            centrality {person.centrality_score ?? "–"} · cluster {clean(person.cluster_id)}
          </span>
        </div>
        <div className="detail-header__actions">
          <Chip className="era-pill" style={{ "--era-color": color } as React.CSSProperties}>
            {eraLabel(person)}
          </Chip>
          <button
            onClick={() =>
              window.postMessage({ type: "FOCUS_PERSON", person_id: personId(person) }, "*")
            }
            type="button"
          >
            View in Constellation
          </button>
        </div>
      </header>

      <section className="detail-section tags-section">
        <h2>Tags and Schools</h2>
        <TagRows label="Math Tags" canonical={person.math_tags} suggested={person.suggested_math_tags} color={color} />
        <TagRows
          label="Cognitive Tags"
          canonical={person.cognitive_tags}
          suggested={person.suggested_cognitive_tags}
          color={color}
        />
        <TagRows
          label="Domain Tags"
          canonical={person.domain_tags}
          suggested={person.suggested_domain_tags}
          color={color}
        />
        <TagRows label="Schools" canonical={person.schools} suggested={person.suggested_schools} color={color} />
      </section>

      {listHas(person.pipeline_flags) || listHas(person.suggested_pipeline_flags) ? (
        <section className="detail-section">
          <h2>Pipeline Flags</h2>
          <div className="chip-row">
            {person.pipeline_flags?.map((flag) => (
              <FlagChip flag={flag} key={flag} />
            ))}
            {person.suggested_pipeline_flags?.map((flag) => (
              <FlagChip flag={flag} key={flag} suggested />
            ))}
          </div>
        </section>
      ) : null}

      {person.pedagogical_significance ? (
        <blockquote className="significance">{person.pedagogical_significance}</blockquote>
      ) : null}

      <section className="detail-section">
        <h2>Relationships ({relationships.length})</h2>
        {relationships.length > 0 ? (
          <div className="relationship-list">
            {relationships.map((rel, index) => {
              const targetName = clean(rel.target);
              const linked = peopleByName.get(targetName.toLowerCase());
              return (
                <RelationshipRow
                  isAiHypothesis={rel.assertion_layer === "ai_hypothesis"}
                  key={`${targetName}-${rel.relation_type}-${index}`}
                  onNavigate={linked ? () => onSelectId(personId(linked)) : undefined}
                  rel={rel}
                />
              );
            })}
          </div>
        ) : (
          <p className="placeholder">No relationships mapped yet</p>
        )}
      </section>

      <section className="detail-section">
        <h2>Bibliography ({bibliography.length} source{bibliography.length !== 1 ? "s" : ""})</h2>
        {bibliography.length > 0 ? (
          bibliography.map((source, index) => (
            <BibliographyCard key={`${source.title ?? "source"}-${index}`} source={source} />
          ))
        ) : (
          <p className="placeholder">No sources indexed yet</p>
        )}
      </section>
    </article>
  );
}

function ReviewMode({
  suggestedTags,
  suggestedPeople,
  aiRelationships,
}: {
  suggestedTags: SuggestedTagEntry[];
  suggestedPeople: SuggestedPersonEntry[];
  aiRelationships: AiRelationshipEntry[];
}) {
  return (
    <div className="review-mode">
      <header className="review-header">
        <p className="eyebrow">Adjudication Queue</p>
        <h2>Review Pipeline Output</h2>
        <p>
          Items below are generated by the indexing pipeline and have not yet been adjudicated. Approve
          to promote to canonical; reject to discard.
        </p>
      </header>

      {suggestedTags.length > 0 ? (
        <section className="review-section">
          <h3>Suggested Tags ({suggestedTags.length})</h3>
          <div className="review-item-list">
            {suggestedTags.slice(0, 200).map((entry, index) => (
              <div className="review-item" key={`tag-${index}`}>
                <div className="review-item__body">
                  <Chip className="chip--suggested">◦ {entry.tag}</Chip>
                  <span className="review-item__meta">{entry.tag_class}</span>
                  {entry.entity_name ? (
                    <span className="review-item__meta">on {entry.entity_name}</span>
                  ) : null}
                  {entry.note ? <p className="review-item__note">{entry.note}</p> : null}
                </div>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      {suggestedPeople.length > 0 ? (
        <section className="review-section">
          <h3>Unmatched People ({suggestedPeople.length})</h3>
          <p className="review-note">
            Names found in source texts that do not match any person in the timeline roster.
          </p>
          <div className="review-item-list">
            {suggestedPeople.slice(0, 200).map((entry, index) => (
              <div className="review-item" key={`person-${index}`}>
                <div className="review-item__body">
                  <strong>{entry.name}</strong>
                  <span className="review-item__meta">{clean(entry.relation)}</span>
                  {entry.source_title ? (
                    <span className="review-item__meta">in {entry.source_title}</span>
                  ) : null}
                  {entry.entity_name ? (
                    <span className="review-item__meta">tagged on {entry.entity_name}</span>
                  ) : null}
                  {entry.note ? <p className="review-item__note">{entry.note}</p> : null}
                </div>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      {aiRelationships.length > 0 ? (
        <section className="review-section">
          <h3>AI Hypothesis Edges ({aiRelationships.length})</h3>
          <p className="review-note">
            Relationships produced by the pipeline with assertion_layer = ai_hypothesis.
            These appear in the Constellation only when "Show AI hypotheses" is enabled.
          </p>
          <div className="review-item-list">
            {aiRelationships.slice(0, 200).map((rel, index) => (
              <div className="review-item review-item--edge" key={`rel-${index}`}>
                <span className="rel-target">{rel.source}</span>
                <span className="rel-arrow">→</span>
                <span className="rel-target">{rel.target}</span>
                <span className="rel-type">{(rel.relation_type ?? "").replace(/_/g, " ")}</span>
                <span className="rel-weight">{rel.weight?.toFixed(2) ?? "–"}</span>
                {rel.confidence ? (
                  <Chip className={`evidence evidence--${evidenceTone[rel.evidence_type ?? ""] ?? "grey"}`}>
                    {rel.confidence}
                  </Chip>
                ) : null}
                {rel.provenance ? <span className="review-item__meta">{rel.provenance}</span> : null}
              </div>
            ))}
          </div>
        </section>
      ) : null}

      {suggestedTags.length === 0 && suggestedPeople.length === 0 && aiRelationships.length === 0 ? (
        <p className="placeholder" style={{ marginTop: "24px" }}>
          No pipeline output to review yet. Run the indexing pipeline to populate this queue.
        </p>
      ) : null}
    </div>
  );
}

type AppMode = "gallery" | "review";

export function ScriptoriumView() {
  const [dataState, setDataState] = useState<TimelineDataState & EnrichedTopLevel>({
    people: [],
    source: "missing",
    awaitingPipeline: true,
  });
  const [isLoading, setIsLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [eraFilter, setEraFilter] = useState<(typeof ERA_TABS)[number]["value"]>("all");
  const [selectedId, setSelectedId] = useState<string>();
  const [mode, setMode] = useState<AppMode>("gallery");

  useEffect(() => {
    loadTimelineData()
      .then((next) => {
        const sorted = [...next.people].sort((left, right) => left.name.localeCompare(right.name));
        setDataState({ ...next, people: sorted });
        setSelectedId(sorted[0] ? personId(sorted[0]) : undefined);
      })
      .finally(() => setIsLoading(false));
  }, []);

  const peopleById = useMemo(
    () => new Map(dataState.people.map((person) => [personId(person), person])),
    [dataState.people],
  );

  // O(1) name-based lookup for relationship navigation.
  const peopleByName = useMemo(
    () => new Map(dataState.people.map((person) => [person.name.toLowerCase(), person])),
    [dataState.people],
  );

  const selectPerson = useCallback(
    (id?: string) => {
      if (id && peopleById.has(id)) {
        setSelectedId(id);
        setMode("gallery");
      }
    },
    [peopleById],
  );

  useEffect(() => {
    const listener = (event: MessageEvent) => {
      if (event.data?.type === "OPEN_PERSON") selectPerson(event.data.person_id as string);
    };
    window.addEventListener("message", listener);
    return () => window.removeEventListener("message", listener);
  }, [selectPerson]);

  const visiblePeople = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return dataState.people.filter((person) => {
      const era = person.calculus_number == null ? "unassigned" : String(person.calculus_number);
      if (eraFilter !== "all" && era !== eraFilter) return false;
      return !needle || person.name.toLowerCase().includes(needle);
    });
  }, [dataState.people, eraFilter, search]);

  const selectedPerson = selectedId ? peopleById.get(selectedId) : undefined;

  const suggestedTags: SuggestedTagEntry[] = dataState.suggested_tags_registry ?? [];
  const suggestedPeople: SuggestedPersonEntry[] = dataState.suggested_people_registry ?? [];
  const aiRelationships: AiRelationshipEntry[] = dataState.ai_relationships ?? [];
  const reviewCount = suggestedTags.length + suggestedPeople.length + aiRelationships.length;

  return (
    <main className="scriptorium-view">
      <aside className="left-panel">
        <header>
          <p className="eyebrow">Genesis Scriptorium</p>
          <h1>Archival Browser</h1>
          <div className="data-status">
            <span>{isLoading ? "Loading…" : `${visiblePeople.length.toLocaleString()} people`}</span>
            <span>
              {dataState.source === "enriched"
                ? "enriched"
                : dataState.source === "fallback"
                  ? "fallback"
                  : "no data"}
            </span>
            {reviewCount > 0 ? <span className="review-badge">{reviewCount} to review</span> : null}
          </div>
        </header>

        <div className="mode-tabs">
          <button
            className={mode === "gallery" ? "is-active" : ""}
            onClick={() => setMode("gallery")}
            type="button"
          >
            Gallery
          </button>
          <button
            className={mode === "review" ? "is-active" : ""}
            onClick={() => setMode("review")}
            type="button"
          >
            Review {reviewCount > 0 ? `(${reviewCount})` : ""}
          </button>
        </div>

        {mode === "gallery" ? (
          <>
            <input
              aria-label="Search people"
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search by name"
              value={search}
            />
            <nav aria-label="Era filter" className="era-tabs">
              {ERA_TABS.map((tab) => (
                <button
                  className={eraFilter === tab.value ? "is-active" : ""}
                  key={tab.value}
                  onClick={() => setEraFilter(tab.value)}
                  type="button"
                >
                  {tab.label}
                </button>
              ))}
            </nav>
            <PersonList
              onSelect={(person) => setSelectedId(personId(person))}
              people={visiblePeople}
              selectedId={selectedId}
            />
          </>
        ) : null}
      </aside>

      {mode === "gallery" ? (
        <DetailView person={selectedPerson} peopleById={peopleById} peopleByName={peopleByName} onSelectId={selectPerson} />
      ) : (
        <ReviewMode
          aiRelationships={aiRelationships}
          suggestedPeople={suggestedPeople}
          suggestedTags={suggestedTags}
        />
      )}
    </main>
  );
}
