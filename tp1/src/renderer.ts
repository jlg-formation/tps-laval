// ---------------------------------------------------------------------------
// Rendu sur le canvas : régions colorées, droite de décision et points.
// ---------------------------------------------------------------------------

import { decisionLineEndpoints, type Vec2 } from "./geometry";
import type { Perceptron } from "./perceptron";
import type { Point } from "./state";

const POINT_RADIUS = 7;

export function pointRadius(): number {
  return POINT_RADIUS;
}

/** Redessine entièrement la scène. */
export function draw(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  points: Point[],
  perceptron: Perceptron,
): void {
  ctx.clearRect(0, 0, width, height);

  drawRegions(ctx, width, height, perceptron);
  drawDecisionLine(ctx, width, height, perceptron);
  drawPoints(ctx, points);
}

/**
 * Teinte légèrement les deux demi-plans (côté prédit « noir » vs « blanc »)
 * pour rendre la frontière lisible. On construit un grand quadrilatère couvrant
 * le demi-plan où la prédiction vaut 1.
 */
function drawRegions(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  perceptron: Perceptron,
): void {
  // Fond = région prédite « blanche ».
  ctx.fillStyle = "rgba(120, 160, 255, 0.10)";
  ctx.fillRect(0, 0, width, height);

  const line = decisionLineEndpoints(perceptron.w1, perceptron.w2, perceptron.b, width, height);
  if (!line) return;

  const [a, c] = line;

  // Direction de la droite et normale pointant vers le côté « prédit = 1 ».
  const dir: Vec2 = { x: c.x - a.x, y: c.y - a.y };
  // Le gradient de la somme pondérée en pixels est proportionnel à (w1, w2) :
  // se déplacer dans ce sens augmente la sortie, donc mène au côté prédit 1.
  const normal: Vec2 = { x: perceptron.w1, y: perceptron.w2 };

  const far = (width + height) * 2;
  const dn = normalize(dir);
  const nn = normalize(normal);
  if (!dn || !nn) return;

  // Quadrilatère : segment étendu de la droite, décalé loin le long de la normale.
  const p1: Vec2 = { x: a.x - dn.x * far, y: a.y - dn.y * far };
  const p2: Vec2 = { x: c.x + dn.x * far, y: c.y + dn.y * far };
  const p3: Vec2 = { x: p2.x + nn.x * far, y: p2.y + nn.y * far };
  const p4: Vec2 = { x: p1.x + nn.x * far, y: p1.y + nn.y * far };

  ctx.fillStyle = "rgba(255, 130, 130, 0.14)";
  ctx.beginPath();
  ctx.moveTo(p1.x, p1.y);
  ctx.lineTo(p2.x, p2.y);
  ctx.lineTo(p3.x, p3.y);
  ctx.lineTo(p4.x, p4.y);
  ctx.closePath();
  ctx.fill();
}

function drawDecisionLine(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  perceptron: Perceptron,
): void {
  const line = decisionLineEndpoints(perceptron.w1, perceptron.w2, perceptron.b, width, height);
  if (!line) return;

  const [a, c] = line;
  ctx.strokeStyle = "#e5484d";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(c.x, c.y);
  ctx.stroke();
}

function drawPoints(ctx: CanvasRenderingContext2D, points: Point[]): void {
  for (const p of points) {
    ctx.beginPath();
    ctx.arc(p.px, p.py, POINT_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = p.label === 1 ? "#101018" : "#ffffff";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#101018";
    ctx.stroke();
  }
}

function normalize(v: Vec2): Vec2 | null {
  const len = Math.hypot(v.x, v.y);
  if (len === 0) return null;
  return { x: v.x / len, y: v.y / len };
}
