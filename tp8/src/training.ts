// Boucle d'interaction agent / environnement, une transition à la fois.
import type { QAgent, QUpdate } from "./agent";
import { step, type Action, type Maze, type Rewards } from "./maze";

export interface EpisodeResult {
  steps: number;
  totalReward: number;
  /** Arrêté par la limite de pas, sans avoir atteint la sortie. */
  truncated: boolean;
}

export interface Transition {
  s: number;
  a: Action;
  r: number;
  next: number;
  terminal: boolean;
  hitWall: boolean;
  update: QUpdate;
  ended?: EpisodeResult;
}

export class Trainer {
  readonly history: EpisodeResult[] = [];
  readonly maxSteps: number;
  position: number;
  steps = 0;
  totalReward = 0;

  constructor(
    readonly maze: Maze,
    readonly agent: QAgent,
    readonly rewards: Rewards,
  ) {
    this.maxSteps = 4 * maze.corridorCount;
    this.position = maze.start;
  }

  tick(): Transition {
    const s = this.position;
    const a = this.agent.chooseAction(s);
    const { next, reward, terminal, hitWall } = step(this.maze, s, a, this.rewards);
    // À la troncature, s' n'est pas terminal : on garde le bootstrap γ max Q(s',·).
    const update = this.agent.update(s, a, reward, next, terminal);
    this.steps++;
    this.totalReward += reward;
    this.position = next;

    let ended: EpisodeResult | undefined;
    if (terminal || this.steps >= this.maxSteps) {
      ended = { steps: this.steps, totalReward: this.totalReward, truncated: !terminal };
      this.history.push(ended);
      this.agent.endEpisode();
      this.position = this.maze.start;
      this.steps = 0;
      this.totalReward = 0;
    }
    return { s, a, r: reward, next, terminal, hitWall, update, ended };
  }

  reset(): void {
    this.agent.reset();
    this.history.length = 0;
    this.position = this.maze.start;
    this.steps = 0;
    this.totalReward = 0;
  }
}
