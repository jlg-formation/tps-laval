export const SIZE = 28;

/** Code latent (z₁, z₂). */
export type Latent = [number, number];

export interface Bounds {
  z1: [number, number];
  z2: [number, number];
}

/** μ d'une image du test et son vrai label. */
export type LatentPoint = [z1: number, z2: number, label: number];

export interface LatentData {
  bounds: Bounds;
  points: LatentPoint[];
}

export const COLORS = [
  "#4e79a7",
  "#f28e2b",
  "#e15759",
  "#76b7b2",
  "#59a14f",
  "#edc948",
  "#b07aa1",
  "#ff9da7",
  "#9c755f",
  "#bab0ac",
];

export async function loadLatentPoints(): Promise<LatentData> {
  const response = await fetch(new URL("latent-points.json", document.baseURI));
  if (!response.ok) {
    throw new Error(`latent-points.json : HTTP ${response.status}`);
  }
  return (await response.json()) as LatentData;
}
