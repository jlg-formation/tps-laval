// Agent Q-learning tabulaire : une valeur Q(s, a) par case et par action.
import type { Action } from "./maze";

export interface AgentParams {
  alpha: number;
  gamma: number;
  epsilon0: number;
  epsMin: number;
  decay: number;
}

export interface QUpdate {
  before: number;
  /** max_a' Q(s', a'), 0 si s' est terminal. */
  maxNext: number;
  target: number;
  after: number;
}

export class QAgent {
  readonly q: Float64Array;
  readonly visited: Uint8Array;
  epsilon: number;

  /** `params` est partagé avec l'interface : une modification s'applique au pas suivant. */
  constructor(
    cellCount: number,
    readonly params: AgentParams,
  ) {
    this.q = new Float64Array(cellCount * 4);
    this.visited = new Uint8Array(cellCount);
    this.epsilon = params.epsilon0;
  }

  /** ε-greedy : action aléatoire avec probabilité ε, sinon la meilleure (égalités tirées au sort). */
  chooseAction(s: number): Action {
    if (Math.random() < this.epsilon) return Math.floor(Math.random() * 4) as Action;
    const base = s * 4;
    let best = -Infinity;
    let choice = 0;
    let ties = 0;
    for (let a = 0; a < 4; a++) {
      const v = this.q[base + a];
      if (v > best) {
        best = v;
        choice = a;
        ties = 1;
      } else if (v === best && Math.random() * ++ties < 1) {
        choice = a;
      }
    }
    return choice as Action;
  }

  maxQ(s: number): number {
    const base = s * 4;
    return Math.max(this.q[base], this.q[base + 1], this.q[base + 2], this.q[base + 3]);
  }

  /** Q(s,a) ← Q(s,a) + α [r + γ max Q(s',·) − Q(s,a)] ; pas de bootstrap si s' est terminal. */
  update(s: number, a: Action, r: number, next: number, terminal: boolean): QUpdate {
    const i = s * 4 + a;
    const before = this.q[i];
    const maxNext = terminal ? 0 : this.maxQ(next);
    const target = r + this.params.gamma * maxNext;
    const after = before + this.params.alpha * (target - before);
    this.q[i] = after;
    this.visited[s] = 1;
    return { before, maxNext, target, after };
  }

  /** Action de la politique gloutonne, ou null si la case n'a rien appris (jamais visitée ou Q toutes égales). */
  greedyAction(s: number): Action | null {
    if (!this.visited[s]) return null;
    const base = s * 4;
    let best = 0;
    for (let a = 1; a < 4; a++) if (this.q[base + a] > this.q[base + best]) best = a;
    const allEqual = this.q[base] === this.q[base + 1] && this.q[base] === this.q[base + 2] && this.q[base] === this.q[base + 3];
    return allEqual ? null : (best as Action);
  }

  endEpisode(): void {
    this.epsilon = Math.max(this.params.epsMin, this.epsilon * this.params.decay);
  }

  reset(): void {
    this.q.fill(0);
    this.visited.fill(0);
    this.epsilon = this.params.epsilon0;
  }
}
