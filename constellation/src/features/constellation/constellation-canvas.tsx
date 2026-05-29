import { useEffect, useMemo, useRef } from "react";
import Graph from "graphology";
import forceAtlas2 from "graphology-layout-forceatlas2";
import Sigma from "sigma";
import { getEdgeColor, getNodeColor } from "./constellation-styles";
import type { GraphEdge, GraphNode } from "./types";

interface ConstellationCanvasProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  visibleNodeIds: ReadonlySet<string>;
  visibleEdgeIds: ReadonlySet<string>;
  dimmedNodeIds: ReadonlySet<string>;
  neighborNodeIds: ReadonlySet<string>;
  hoveredNodeId?: string;
  selectedNodeId?: string;
  onHoverNodeChange: (nodeId?: string, position?: { x: number; y: number }) => void;
  onSelectNode: (nodeId?: string) => void;
}

const assignCircularPositions = (graph: Graph, nodes: GraphNode[]) => {
  const clusters = Array.from(
    nodes.reduce((map, node) => {
      const key = node.cluster_id || node.field || String(node.calculus_number ?? "unknown");
      map.set(key, [...(map.get(key) ?? []), node]);
      return map;
    }, new Map<string, GraphNode[]>()),
  ).sort((left, right) => right[1].length - left[1].length);

  clusters.forEach(([, clusterNodes], clusterIndex) => {
    const clusterAngle = (clusterIndex / Math.max(clusters.length, 1)) * Math.PI * 2;
    const centerRadius = nodes.length > 80 ? 3.4 : 1.5;
    const centerX = Math.cos(clusterAngle) * centerRadius;
    const centerY = Math.sin(clusterAngle) * centerRadius;
    clusterNodes.forEach((node, nodeIndex) => {
      const angle = (nodeIndex / Math.max(clusterNodes.length, 1)) * Math.PI * 2;
      const radius = 0.18 + Math.sqrt(clusterNodes.length) * 0.05;
      graph.setNodeAttribute(node.id, "x", centerX + Math.cos(angle) * radius);
      graph.setNodeAttribute(node.id, "y", centerY + Math.sin(angle) * radius);
    });
  });
};

const assignLayoutPositions = (graph: Graph, nodes: GraphNode[]) => {
  assignCircularPositions(graph, nodes);
  if (nodes.length < 2) return;

  try {
    forceAtlas2.assign(graph, {
      iterations: nodes.length > 1200 ? 90 : 160,
      settings: {
        ...forceAtlas2.inferSettings(graph),
        gravity: 0.08,
        scalingRatio: 18,
        slowDown: 8,
        barnesHutOptimize: true,
      },
    });
  } catch {
    assignCircularPositions(graph, nodes);
  }
};

const makeNodeReducer =
  (
    nodeLookup: Map<string, GraphNode>,
    dimmedNodeIds: ReadonlySet<string>,
    neighborNodeIds: ReadonlySet<string>,
    hoveredNodeId?: string,
    selectedNodeId?: string,
  ) =>
  (nodeKey: string, data: Record<string, unknown>) => {
    const node = nodeLookup.get(nodeKey);
    if (!node) return data;
    const isHovered = hoveredNodeId === nodeKey;
    const isSelected = selectedNodeId === nodeKey;
    const isNeighbor = neighborNodeIds.has(nodeKey);
    const isDimmed = dimmedNodeIds.has(nodeKey);
    const isFocused = isSelected || isNeighbor || isHovered;
    const enriched: GraphNode = { ...node, isDimmed, isNeighbor, isSelected };
    return {
      ...data,
      color: getNodeColor(enriched),
      size: Math.max(node.size, isFocused ? node.size + 3 : node.size),
      label: isFocused || node.degree > 3 || node.isSearchMatch ? node.label : "",
      zIndex: isFocused ? 4 : 1,
      highlighted: isFocused,
    };
  };

const makeEdgeReducer =
  (edgeLookup: Map<string, GraphEdge>, selectedNodeId?: string, hoveredNodeId?: string) =>
  (edgeKey: string, data: Record<string, unknown>) => {
    const edge = edgeLookup.get(edgeKey);
    if (!edge) return data;
    const isSelectionAdjacent = Boolean(
      selectedNodeId && (edge.source === selectedNodeId || edge.target === selectedNodeId),
    );
    const isDimmed = Boolean(selectedNodeId && !isSelectionAdjacent);
    const isHoveredAdjacent = Boolean(
      hoveredNodeId && (edge.source === hoveredNodeId || edge.target === hoveredNodeId),
    );
    const enriched: GraphEdge = { ...edge, isSelectionAdjacent, isDimmed };
    return {
      ...data,
      color: isHoveredAdjacent ? edge.color : getEdgeColor(enriched),
      size: isSelectionAdjacent || isHoveredAdjacent ? 2.2 : Math.max(0.4, edge.weight * 1.4),
      zIndex: isSelectionAdjacent || isHoveredAdjacent ? 2 : 0,
    };
  };

export function ConstellationCanvas({
  nodes,
  edges,
  visibleNodeIds,
  visibleEdgeIds,
  dimmedNodeIds,
  neighborNodeIds,
  hoveredNodeId,
  selectedNodeId,
  onHoverNodeChange,
  onSelectNode,
}: ConstellationCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const sigmaRef = useRef<Sigma | null>(null);

  const nodeLookup = useMemo(() => new Map(nodes.map((node) => [node.id, node])), [nodes]);
  const edgeLookup = useMemo(() => new Map(edges.map((edge) => [edge.id, edge])), [edges]);

  // Effect 1: full rebuild when graph structure changes (runs once on data load).
  // ForceAtlas2 runs here — not on filter toggles.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const graph = new Graph({ multi: true, type: "directed" });
    nodes.forEach((node) => {
      graph.mergeNode(node.id, {
        x: 0,
        y: 0,
        size: node.size,
        color: node.color,
        label: node.label,
        type: "circle",
        forceLabel: false,
        hidden: !visibleNodeIds.has(node.id),
      });
    });

    assignLayoutPositions(graph, nodes);

    edges.forEach((edge) => {
      if (!graph.hasNode(edge.source) || !graph.hasNode(edge.target)) return;
      graph.mergeEdgeWithKey(edge.id, edge.source, edge.target, {
        size: Math.max(0.4, edge.weight * 1.4),
        color: edge.color,
        type: "line",
        label: edge.relation_type,
        weight: edge.weight,
        hidden: !visibleEdgeIds.has(edge.id),
      });
    });

    const sigma = new Sigma(graph, container, {
      allowInvalidContainer: true,
      defaultEdgeType: "line",
      labelDensity: 0.08,
      labelRenderedSizeThreshold: 9,
      renderEdgeLabels: false,
      zIndex: true,
      nodeReducer: makeNodeReducer(nodeLookup, dimmedNodeIds, neighborNodeIds, hoveredNodeId, selectedNodeId),
      edgeReducer: makeEdgeReducer(edgeLookup, selectedNodeId, hoveredNodeId),
    });

    sigma.on("enterNode", ({ node }) => {
      const display = sigma.getNodeDisplayData(node);
      onHoverNodeChange(node, display ? { x: display.x, y: display.y } : undefined);
    });
    sigma.on("leaveNode", () => onHoverNodeChange(undefined));
    sigma.on("clickNode", ({ node }) => onSelectNode(node));
    sigma.on("clickStage", () => onSelectNode(undefined));

    sigmaRef.current = sigma;
    return () => {
      sigma.kill();
      sigmaRef.current = null;
    };
    // Only rebuild when graph structure changes — intentionally omit visibility/emphasis deps.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes, edges, onHoverNodeChange, onSelectNode]);

  // Effect 2: update hidden attributes on filter toggle (no layout recalculation).
  useEffect(() => {
    const sigma = sigmaRef.current;
    if (!sigma) return;
    const graph = sigma.getGraph();
    graph.forEachNode((nodeId) => {
      graph.setNodeAttribute(nodeId, "hidden", !visibleNodeIds.has(nodeId));
    });
    graph.forEachEdge((edgeId) => {
      graph.setEdgeAttribute(edgeId, "hidden", !visibleEdgeIds.has(edgeId));
    });
    sigma.refresh();
  }, [visibleNodeIds, visibleEdgeIds]);

  // Effect 3: update visual emphasis on hover/selection (no layout, no hidden changes).
  useEffect(() => {
    const sigma = sigmaRef.current;
    if (!sigma) return;
    sigma.setSetting(
      "nodeReducer",
      makeNodeReducer(nodeLookup, dimmedNodeIds, neighborNodeIds, hoveredNodeId, selectedNodeId),
    );
    sigma.setSetting("edgeReducer", makeEdgeReducer(edgeLookup, selectedNodeId, hoveredNodeId));
    sigma.refresh();
  }, [nodeLookup, edgeLookup, dimmedNodeIds, neighborNodeIds, hoveredNodeId, selectedNodeId]);

  return <div className="constellation-canvas" ref={containerRef} />;
}
