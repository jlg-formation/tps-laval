// Boucle d'interaction agent ↔ environnement : épisodes, replay buffer, apprentissage, baseline aléatoire.

import { ACTIONS, Breakout, OBS_SIZE, type Action, type StepResult } from "./breakout";
import { LOSS_EVERY, type DqnAgent } from "./dqn";
import { ReplayBuffer } from "./replay";

/** Répétition d'action (frame skip) : l'agent décide un pas de physique sur ACTION_REPEAT. */
export const ACTION_REPEAT = 4;
export const BUFFER_SIZE = 50_000;
export const BATCH_SIZE = 64;
export const WARMUP = 1_000;
/** Une descente de gradient toutes les TRAIN_EVERY décisions (4 comme le DQN Atari) : 4× moins de calcul. */
export const TRAIN_EVERY = 4;
const LOSS_POINTS = 1_000;

export type Mode = "train" | "demo" | "human";

export interface EpisodeResult {
  mode: Mode;
  score: number;
  steps: number;
  truncated: boolean;
}

/** Historique de la perte à taille bornée : quand il est plein, on moyenne les points deux à deux. */
export class LossHistory {
  readonly values: number[] = [];
  /** Mises à jour du réseau couvertes par un point. */
  updatesPerPoint = LOSS_EVERY;
  private sum = 0;
  private count = 0;
  private samplesPerPoint = 1;

  add(loss: number): void {
    this.sum += loss;
    if (++this.count < this.samplesPerPoint) return;
    this.values.push(this.sum / this.count);
    this.sum = 0;
    this.count = 0;
    if (this.values.length < LOSS_POINTS) return;
    for (let i = 0; i < this.values.length / 2; i++) this.values[i] = (this.values[2 * i] + this.values[2 * i + 1]) / 2;
    this.values.length = LOSS_POINTS / 2;
    this.samplesPerPoint *= 2;
    this.updatesPerPoint *= 2;
  }

  clear(): void {
    this.values.length = 0;
    this.sum = 0;
    this.count = 0;
    this.samplesPerPoint = 1;
    this.updatesPerPoint = LOSS_EVERY;
  }
}

export class Trainer {
  readonly env = new Breakout();
  readonly buffer = new ReplayBuffer(BUFFER_SIZE, OBS_SIZE, BATCH_SIZE);
  /** Épisodes d'entraînement uniquement (la démo et le jeu humain n'y figurent pas). */
  readonly history: EpisodeResult[] = [];
  readonly losses = new LossHistory();
  mode: Mode = "train";
  /** Action tenue par le joueur humain (clavier ou souris). */
  humanAction: Action = 1;
  /** Faux en turbo : on saute le passage avant quand ε impose une action au hasard. */
  wantQ = true;
  /** Q-valeurs de la dernière décision, pour l'affichage. */
  readonly q = new Float32Array(ACTIONS.length);
  hasQ = false;
  action: Action = 1;
  randomAction = false;
  lastEpisode: EpisodeResult | null = null;
  private readonly obs = new Float32Array(OBS_SIZE);
  private readonly nextObs = new Float32Array(OBS_SIZE);
  private repeatLeft = 0;
  private reward = 0;

  constructor(readonly agent: DqnAgent) {}

  setMode(mode: Mode): void {
    this.mode = mode;
    this.newGame();
  }

  newGame(): void {
    this.env.reset();
    this.repeatLeft = 0;
    this.hasQ = false;
  }

  /** Avance d'un pas de physique ; renvoie le résultat de l'épisode s'il vient de se terminer. */
  tick(): EpisodeResult | null {
    let r: StepResult;
    if (this.mode === "human") {
      this.action = this.humanAction;
      r = this.env.step(this.action);
    } else {
      if (this.repeatLeft === 0) this.decide();
      r = this.env.step(this.action);
      this.reward += r.reward;
      this.repeatLeft--;
      if (this.repeatLeft === 0 || r.lifeLost || r.terminal || r.truncated) {
        this.repeatLeft = 0;
        if (this.mode === "train") this.learnFrom(r);
      }
    }
    if (!r.terminal && !r.truncated) return null;

    const result: EpisodeResult = { mode: this.mode, score: this.env.score, steps: this.env.steps, truncated: r.truncated };
    if (this.mode === "train") this.history.push(result);
    this.lastEpisode = result;
    this.newGame();
    return result;
  }

  private decide(): void {
    this.env.observe(this.obs);
    const epsilon = this.mode === "train" ? this.agent.epsilon : 0;
    const choice = this.agent.act(this.obs, epsilon, this.q, this.wantQ);
    this.action = choice.action;
    this.randomAction = choice.random;
    this.hasQ = choice.hasQ;
    this.repeatLeft = ACTION_REPEAT;
    this.reward = 0;
  }

  private learnFrom(r: StepResult): void {
    this.env.observe(this.nextObs);
    // Vie perdue = état terminal pour la cible (astuce DQN Atari) ; la troncature, elle, garde γ·max Q(s', ·).
    this.buffer.push(this.obs, this.action, this.reward, this.nextObs, r.lifeLost || r.terminal);
    this.agent.countStep();
    if (this.buffer.size < WARMUP || this.agent.steps % TRAIN_EVERY !== 0) return;
    this.agent.learn(this.buffer.sample());
    if (this.agent.updates % LOSS_EVERY === 0) this.losses.add(this.agent.lastLoss);
  }

  /** Q-valeurs de l'agent dans l'état courant, sans agir (mode humain). */
  peekQ(): void {
    this.env.observe(this.obs);
    this.agent.qValues(this.obs, this.q);
    this.hasQ = true;
  }

  /** Oublie l'expérience passée : à appeler après un chargement de poids. */
  clearBuffer(): void {
    this.buffer.clear();
  }

  reset(): void {
    this.agent.reset();
    this.buffer.clear();
    this.history.length = 0;
    this.losses.clear();
    this.lastEpisode = null;
    this.newGame();
  }
}

/** Score moyen d'une politique uniformément aléatoire (même répétition d'action que l'agent). */
export function randomBaseline(games = 30): number {
  const env = new Breakout();
  let total = 0;
  for (let g = 0; g < games; g++) {
    env.reset();
    let action: Action = 1;
    let left = 0;
    for (;;) {
      if (left === 0) {
        action = ACTIONS[Math.floor(Math.random() * ACTIONS.length)];
        left = ACTION_REPEAT;
      }
      const r = env.step(action);
      left = r.lifeLost ? 0 : left - 1;
      if (r.terminal || r.truncated) break;
    }
    total += env.score;
  }
  return total / games;
}
