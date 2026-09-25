// Environnement : un labyrinthe sur une grille, états = cases, 4 actions déterministes.

export type Action = 0 | 1 | 2 | 3;
export const ACTIONS: readonly Action[] = [0, 1, 2, 3];
export const ACTION_NAMES = ["haut", "bas", "gauche", "droite"] as const;
export const ACTION_ARROWS = ["↑", "↓", "←", "→"] as const;
export const DX = [0, 0, -1, 1] as const;
export const DY = [-1, 1, 0, 0] as const;

export interface Maze {
  /** Nombre de cellules de labyrinthe par côté. */
  n: number;
  /** Côté de la grille affichée, murs compris : 2n + 1. */
  width: number;
  /** 1 = mur, 0 = couloir, indexé par y * width + x. */
  walls: Uint8Array;
  start: number;
  goal: number;
  corridorCount: number;
}

export interface Rewards {
  goal: number;
  step: number;
  wall: number;
}

export interface StepResult {
  next: number;
  reward: number;
  terminal: boolean;
  hitWall: boolean;
}

/** Labyrinthe « parfait » (un seul chemin entre deux cases) par recursive backtracker. */
export function generateMaze(n: number): Maze {
  const width = 2 * n + 1;
  const walls = new Uint8Array(width * width).fill(1);
  const seen = new Uint8Array(n * n);
  const open = (cx: number, cy: number) => {
    walls[(2 * cy + 1) * width + 2 * cx + 1] = 0;
  };

  // Pile explicite : une récursion ferait déborder la pile sur les grands labyrinthes.
  const stack = [0];
  seen[0] = 1;
  open(0, 0);
  const candidates: Action[] = [];
  while (stack.length > 0) {
    const cur = stack[stack.length - 1];
    const cx = cur % n;
    const cy = Math.floor(cur / n);
    candidates.length = 0;
    for (const a of ACTIONS) {
      const nx = cx + DX[a];
      const ny = cy + DY[a];
      if (nx >= 0 && nx < n && ny >= 0 && ny < n && !seen[ny * n + nx]) candidates.push(a);
    }
    if (candidates.length === 0) {
      stack.pop();
      continue;
    }
    const a = candidates[Math.floor(Math.random() * candidates.length)];
    const nx = cx + DX[a];
    const ny = cy + DY[a];
    walls[(2 * cy + 1 + DY[a]) * width + 2 * cx + 1 + DX[a]] = 0;
    open(nx, ny);
    seen[ny * n + nx] = 1;
    stack.push(ny * n + nx);
  }

  return {
    n,
    width,
    walls,
    start: width + 1,
    goal: (width - 2) * width + (width - 2),
    // n² cellules + (n² − 1) murs percés (arbre couvrant).
    corridorCount: 2 * n * n - 1,
  };
}

/** Longueur (en pas) du plus court chemin départ → sortie, par parcours en largeur. */
export function shortestPathLength(maze: Maze): number {
  const { width, walls, start, goal } = maze;
  const dist = new Int32Array(width * width).fill(-1);
  dist[start] = 0;
  const queue = [start];
  for (let head = 0; head < queue.length; head++) {
    const s = queue[head];
    if (s === goal) return dist[s];
    for (const a of ACTIONS) {
      const next = s + DY[a] * width + DX[a];
      if (!walls[next] && dist[next] < 0) {
        dist[next] = dist[s] + 1;
        queue.push(next);
      }
    }
  }
  return -1;
}

/** Dynamique de l'environnement. La bordure est toujours un mur : on ne sort jamais de la grille. */
export function step(maze: Maze, s: number, a: Action, rewards: Rewards): StepResult {
  const next = s + DY[a] * maze.width + DX[a];
  if (maze.walls[next]) return { next: s, reward: rewards.wall, terminal: false, hitWall: true };
  if (next === maze.goal) return { next, reward: rewards.goal, terminal: true, hitWall: false };
  return { next, reward: rewards.step, terminal: false, hitWall: false };
}
