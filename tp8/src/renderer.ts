// Dessin du labyrinthe : murs, heatmap de V(s) = max Q, flèches de la politique, agent.
import type { QAgent } from "./agent";
import { DX, DY, type Maze } from "./maze";

const MAX_PX = 600;
const WALL = "#11111b";
const CORRIDOR = "#33334a";
const START = "#9ece6a";
const GOAL = "#e0af68";
const AGENT = "#f7768e";

// Palette « viridis » simplifiée : faible → violet foncé, élevé → jaune.
const PALETTE: readonly [number, number, number][] = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37],
];

export const PALETTE_GRADIENT = `linear-gradient(to right, ${PALETTE.map(([r, g, b]) => `rgb(${r}, ${g}, ${b})`).join(", ")})`;

function paletteColor(t: number): string {
  const x = Math.min(1, Math.max(0, t)) * (PALETTE.length - 1);
  const i = Math.min(PALETTE.length - 2, Math.floor(x));
  const f = x - i;
  const [r0, g0, b0] = PALETTE[i];
  const [r1, g1, b1] = PALETTE[i + 1];
  return `rgb(${Math.round(r0 + (r1 - r0) * f)}, ${Math.round(g0 + (g1 - g0) * f)}, ${Math.round(b0 + (b1 - b0) * f)})`;
}

export interface ValueRange {
  min: number;
  max: number;
}

export class MazeRenderer {
  private readonly ctx: CanvasRenderingContext2D;
  private maze: Maze | null = null;
  private cell = 0;

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
  }

  setMaze(maze: Maze): void {
    this.maze = maze;
    this.cell = Math.floor(MAX_PX / maze.width);
    const px = this.cell * maze.width;
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = px * dpr;
    this.canvas.height = px * dpr;
    this.canvas.style.width = `${px}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  cellAt(event: MouseEvent): number | null {
    if (!this.maze) return null;
    const rect = this.canvas.getBoundingClientRect();
    const x = Math.floor(((event.clientX - rect.left) / rect.width) * this.maze.width);
    const y = Math.floor(((event.clientY - rect.top) / rect.height) * this.maze.width);
    if (x < 0 || y < 0 || x >= this.maze.width || y >= this.maze.width) return null;
    return y * this.maze.width + x;
  }

  /** Renvoie l'intervalle de V utilisé pour la heatmap, ou null si aucune case n'a encore été visitée. */
  draw(agent: QAgent, position: number, hover: number | null): ValueRange | null {
    const maze = this.maze;
    if (!maze) return null;
    const { ctx, cell: c } = this;
    const { width, walls } = maze;

    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < walls.length; i++) {
      if (walls[i] || !agent.visited[i]) continue;
      const v = agent.maxQ(i);
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max >= min ? { min, max } : null;
    const norm = (v: number) => (range && max > min ? (v - min) / (max - min) : 0.5);

    for (let i = 0; i < walls.length; i++) {
      const x = (i % width) * c;
      const y = Math.floor(i / width) * c;
      if (walls[i]) ctx.fillStyle = WALL;
      else if (i === maze.goal) ctx.fillStyle = GOAL;
      else if (agent.visited[i]) ctx.fillStyle = paletteColor(norm(agent.maxQ(i)));
      else ctx.fillStyle = CORRIDOR;
      ctx.fillRect(x, y, c, c);
    }

    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (let i = 0; i < walls.length; i++) {
      if (walls[i]) continue;
      const a = agent.greedyAction(i);
      if (a === null) continue;
      const t = norm(agent.maxQ(i));
      this.drawArrow(i, DX[a], DY[a], t > 0.6 ? "rgba(17, 17, 27, 0.85)" : "rgba(255, 255, 255, 0.85)");
    }

    this.outlineCell(maze.start, START, Math.max(2, c * 0.12));
    if (c >= 14) {
      this.label(maze.start, "S", START);
      this.label(maze.goal, "G", WALL);
    }

    const ax = (position % width) * c + c / 2;
    const ay = Math.floor(position / width) * c + c / 2;
    ctx.fillStyle = AGENT;
    ctx.strokeStyle = WALL;
    ctx.lineWidth = Math.max(1, c * 0.06);
    ctx.beginPath();
    ctx.arc(ax, ay, c * 0.3, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    if (hover !== null) this.outlineCell(hover, "#ffffff", 1.5);
    return range;
  }

  private drawArrow(i: number, dx: number, dy: number, color: string): void {
    const { ctx, cell: c } = this;
    const width = this.maze!.width;
    const cx = (i % width) * c + c / 2;
    const cy = Math.floor(i / width) * c + c / 2;
    const len = c * 0.3;
    const head = c * 0.18;
    const tx = cx + dx * len;
    const ty = cy + dy * len;
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(1, c * 0.08);
    ctx.beginPath();
    ctx.moveTo(cx - dx * len, cy - dy * len);
    ctx.lineTo(tx, ty);
    // Pointe : deux segments à ±45° vers l'arrière.
    ctx.moveTo(tx - dx * head - dy * head, ty - dy * head - dx * head);
    ctx.lineTo(tx, ty);
    ctx.lineTo(tx - dx * head + dy * head, ty - dy * head + dx * head);
    ctx.stroke();
  }

  private outlineCell(i: number, color: string, lineWidth: number): void {
    const { ctx, cell: c } = this;
    const width = this.maze!.width;
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.strokeRect((i % width) * c + lineWidth / 2, Math.floor(i / width) * c + lineWidth / 2, c - lineWidth, c - lineWidth);
  }

  private label(i: number, text: string, color: string): void {
    const { ctx, cell: c } = this;
    const width = this.maze!.width;
    ctx.fillStyle = color;
    ctx.font = `bold ${Math.round(c * 0.32)}px system-ui, sans-serif`;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(text, (i % width) * c + c * 0.1, Math.floor(i / width) * c + c * 0.06);
  }
}
