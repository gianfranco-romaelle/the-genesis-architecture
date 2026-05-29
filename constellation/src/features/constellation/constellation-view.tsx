import { useCallback, useEffect, useMemo, useState } from "react";
import { ConstellationCanvas } from "./constellation-canvas";
import { calculusColor } from "./constellation-styles";
import { loadTimelineData } from "./data-loader";
import { EntityCard } from "./entity-card";
import { FilterPanel } from "./filter-panel";
import { buildGraph, personId } from "./graph-builder";
import { ScriptoriumSidebar } from "./scriptorium-sidebar";
import type { TimelineDataState } from "./types";

const sameValue = (left: number | null, right: number | null) => left === right;

const toggleArrayValue = <T,>(values: T[], value: T, compare: (left: T, right: T) => boolean = Object.is) =>
  values.some((item) => compare(item, value))
    ? values.filter((item) => !compare(item, value))
    : [...values, value];

export function ConstellationView() {
  const [dataState, setDataState] = useState<TimelineDataState>({
    people: [],
    source: "fallback",
    awaitingPipeline: true,
  });
  const [isLoading, setIsLoading] = useState(true);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | undefined>();
  const [hoverPosition, setHoverPosition] = useState<{ x: number; y: number } | undefined>();
  const [selectedNodeId, setSelectedNodeId] = useState<string | undefined>();
  const [enabledCalculi, setEnabledCalculi] = useState<Array<number | null>>([]);
  const [enabledRelations, setEnabledRelations] = useState<string[]>([]);
  const [selectedSchools, setSelectedSchools] = useState<string[]>([]);
  const [minimumWeight, setMinimumWeight] = useState(0.3);
  const [showAiHypotheses, setShowAiHypotheses] = useState(false);
  const [scriptoriumPersonId, setScriptoriumPersonId] = useState<string | undefined>();

  useEffect(() => {
    loadTimelineData()
      .then((nextData) => {
        setDataState(nextData);
        const calculi = Array.from(new Set(nextData.people.map((person) => person.calculus_number ?? null)));
        setEnabledCalculi(calculi);
        const relationTypes = Array.from(
          new Set(
            nextData.people.flatMap(
              (person) => person.semantic_relationships?.map((rel) => rel.relation_type) ?? [],
            ),
          ),
        );
        setEnabledRelations(relationTypes);
      })
      .finally(() => setIsLoading(false));
  }, []);

  // Full graph built once — ForceAtlas2 layout runs inside the canvas, not here.
  const baseGraph = useMemo(
    () => buildGraph(dataState.people, { showAiHypotheses }),
    [dataState.people, showAiHypotheses],
  );

  const filterOptions = useMemo(() => {
    const calculi = Array.from(
      new Set(dataState.people.map((person) => person.calculus_number ?? null)),
    ).sort((left, right) => {
      if (left === null) return 1;
      if (right === null) return -1;
      return left - right;
    });
    const relationTypes = Array.from(new Set(baseGraph.edges.map((edge) => edge.relation_type))).sort();
    const schools = Array.from(
      new Set(dataState.people.flatMap((person) => [...(person.schools ?? []), ...(person.suggested_schools ?? [])])),
    ).sort();
    return { calculi, relationTypes, schools };
  }, [baseGraph.edges, dataState.people]);

  // Which nodes pass the calculus/school filter.
  const visibleNodeIds = useMemo(() => {
    const schools = new Set(selectedSchools);
    return new Set(
      baseGraph.nodes
        .filter((node) => enabledCalculi.some((c) => sameValue(c, node.calculus_number ?? null)))
        .filter((node) => {
          if (schools.size === 0) return true;
          return [...(node.schools ?? []), ...(node.suggested_schools ?? [])].some((s) => schools.has(s));
        })
        .map((node) => node.id),
    );
  }, [baseGraph.nodes, enabledCalculi, selectedSchools]);

  // Which edges pass the type/weight filter and have both endpoints visible.
  const visibleEdgeIds = useMemo(() => {
    const relationSet = new Set(enabledRelations);
    return new Set(
      baseGraph.edges
        .filter(
          (edge) =>
            visibleNodeIds.has(edge.source) &&
            visibleNodeIds.has(edge.target) &&
            edge.weight >= minimumWeight &&
            (relationSet.size === 0 || relationSet.has(edge.relation_type)),
        )
        .map((edge) => edge.id),
    );
  }, [baseGraph.edges, visibleNodeIds, enabledRelations, minimumWeight]);

  // Neighbors of the selected node among visible edges.
  const neighborNodeIds = useMemo(() => {
    if (!selectedNodeId) return new Set<string>();
    const neighbors = new Set<string>();
    baseGraph.edges.forEach((edge) => {
      if (!visibleEdgeIds.has(edge.id)) return;
      if (edge.source === selectedNodeId) neighbors.add(edge.target);
      if (edge.target === selectedNodeId) neighbors.add(edge.source);
    });
    return neighbors;
  }, [selectedNodeId, baseGraph.edges, visibleEdgeIds]);

  // Visible nodes that are neither selected nor neighbors — rendered dim.
  const dimmedNodeIds = useMemo(() => {
    if (!selectedNodeId) return new Set<string>();
    return new Set(
      Array.from(visibleNodeIds).filter((id) => id !== selectedNodeId && !neighborNodeIds.has(id)),
    );
  }, [selectedNodeId, visibleNodeIds, neighborNodeIds]);

  // Only focus a node if it is currently visible.
  const focusNode = useCallback(
    (nodeId?: string) => {
      if (!nodeId || visibleNodeIds.has(nodeId)) {
        setSelectedNodeId(nodeId);
      }
    },
    [visibleNodeIds],
  );

  const focusPersonEverywhere = useCallback(
    (nodeId: string) => {
      const person = dataState.people.find((item) => personId(item) === nodeId);
      if (!person) return;
      setSelectedNodeId(nodeId);
      setScriptoriumPersonId(nodeId);
    },
    [dataState.people],
  );

  useEffect(() => {
    const listener = (event: MessageEvent) => {
      if (event.data?.type === "FOCUS_PERSON") {
        focusNode(event.data.person_id as string);
      }
    };
    window.addEventListener("message", listener);
    return () => window.removeEventListener("message", listener);
  }, [focusNode]);

  // Hover/selected node — look up from baseGraph so EntityCard always has full data.
  const activeNode = baseGraph.nodes.find((node) => node.id === (hoveredNodeId ?? selectedNodeId));

  const resetFilters = () => {
    setEnabledCalculi(filterOptions.calculi);
    setEnabledRelations(filterOptions.relationTypes);
    setSelectedSchools([]);
    setMinimumWeight(0.3);
    setShowAiHypotheses(false);
  };

  return (
    <main className="constellation-view">
      <ConstellationCanvas
        nodes={baseGraph.nodes}
        edges={baseGraph.edges}
        visibleNodeIds={visibleNodeIds}
        visibleEdgeIds={visibleEdgeIds}
        dimmedNodeIds={dimmedNodeIds}
        neighborNodeIds={neighborNodeIds}
        hoveredNodeId={hoveredNodeId}
        selectedNodeId={selectedNodeId}
        onHoverNodeChange={(nodeId, position) => {
          setHoveredNodeId(nodeId);
          setHoverPosition(position);
        }}
        onSelectNode={focusNode}
      />

      <FilterPanel
        calculi={filterOptions.calculi}
        relationTypes={filterOptions.relationTypes}
        schools={filterOptions.schools}
        enabledCalculi={enabledCalculi}
        enabledRelations={enabledRelations}
        selectedSchools={selectedSchools}
        minimumWeight={minimumWeight}
        showAiHypotheses={showAiHypotheses}
        onToggleCalculus={(value) => setEnabledCalculi((current) => toggleArrayValue(current, value, sameValue))}
        onToggleRelation={(value) => setEnabledRelations((current) => toggleArrayValue(current, value))}
        onToggleSchool={(value) => setSelectedSchools((current) => toggleArrayValue(current, value))}
        onMinimumWeightChange={setMinimumWeight}
        onToggleAiHypotheses={() => setShowAiHypotheses((current) => !current)}
        onReset={resetFilters}
      />

      <div className="status-strip">
        <span>{isLoading ? "Loading" : `${visibleNodeIds.size.toLocaleString()} figures`}</span>
        <span>{visibleEdgeIds.size.toLocaleString()} relations</span>
        <span>{dataState.source === "enriched" ? "enriched data" : "fallback data"}</span>
        {showAiHypotheses ? <span className="ai-badge">AI hypotheses ON</span> : null}
      </div>

      {dataState.awaitingPipeline ? (
        <div className="pipeline-badge">Pipeline output pending</div>
      ) : null}

      {activeNode ? (
        <EntityCard
          person={activeNode}
          position={hoverPosition}
          onOpenScriptorium={(person) => {
            setSelectedNodeId(person.id);
            setScriptoriumPersonId(person.id);
          }}
        />
      ) : null}

      {visibleNodeIds.size === 0 && !isLoading ? (
        <div className="empty-state">
          <p className="eyebrow">Constellation</p>
          <h2>No visible figures</h2>
          <button type="button" onClick={resetFilters}>
            Reset filters
          </button>
        </div>
      ) : null}

      <div className="calculus-legend">
        {filterOptions.calculi.map((value) => (
          <span key={String(value)}>
            <i style={{ backgroundColor: calculusColor(value) }} />
            {value === null ? "Unknown" : `Era ${value}`}
          </span>
        ))}
      </div>

      <ScriptoriumSidebar
        person={scriptoriumPersonId ? dataState.people.find((person) => personId(person) === scriptoriumPersonId) : undefined}
        people={dataState.people}
        onClose={() => setScriptoriumPersonId(undefined)}
        onFocusPerson={focusPersonEverywhere}
      />
    </main>
  );
}
