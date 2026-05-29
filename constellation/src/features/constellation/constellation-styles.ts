import type { GraphEdge, GraphNode } from "./types";

export const CALCULUS_COLORS: Record<string, string> = {
  "0": "#c9a96e",
  "1": "#7ec8c8",
  "2": "#a8d8a8",
  "3": "#b8a0d8",
  "4": "#f0a060",
  "5": "#80b8f0",
  "6": "#d4a0c8",
  null: "#888888",
};

export const RELATION_COLORS: Record<string, string> = {
  influenced: "#7ec8c8",
  collaborated_with: "#a8d8a8",
  corresponded_with: "#80b8f0",
  taught: "#c9a96e",
  cited: "#b8a0d8",
  debated: "#f0a060",
  opposed: "#d97878",
  associated_with: "#9a8f79",
};

export const calculusColor = (calculusNumber: number | null | undefined) =>
  CALCULUS_COLORS[String(calculusNumber ?? "null")] ?? "#888888";

export const relationColor = (relationType: string) =>
  RELATION_COLORS[relationType] ?? "#ffffff44";

export const dimHex = (hex: string) => {
  if (!hex.startsWith("#")) return hex;
  return hex.length === 7 ? `${hex}33` : hex;
};

export const getNodeColor = (node: GraphNode) =>
  node.isDimmed ? dimHex(node.color) : node.color;

export const getEdgeColor = (edge: GraphEdge) =>
  edge.isDimmed ? dimHex(edge.color) : edge.color;
