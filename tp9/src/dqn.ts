// Agent DQN : réseau Q, réseau cible, politique ε-greedy et pas d'apprentissage (perte de Huber, Adam).

import * as tf from "@tensorflow/tfjs";
import { ACTIONS, OBS_SIZE, type Action } from "./breakout";
import type { Batch } from "./replay";

export interface DqnParams {
  learningRate: number;
  gamma: number;
  epsStart: number;
  epsEnd: number;
  /** Nombre de pas pour passer linéairement de ε initial à ε final. */
  epsDecaySteps: number;
  /** Période C (en pas) de recopie du réseau Q vers le réseau cible. */
  targetPeriod: number;
}

const HIDDEN = 128;
const N_ACTIONS = ACTIONS.length;
// Lire la perte force une synchronisation GPU → CPU : on ne le fait qu'une mise à jour sur LOSS_EVERY.
export const LOSS_EVERY = 10;

function buildQNetwork(): tf.Sequential {
  return tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [OBS_SIZE], units: HIDDEN, activation: "relu" }),
      tf.layers.dense({ units: HIDDEN, activation: "relu" }),
      tf.layers.dense({ units: N_ACTIONS }),
    ],
  });
}

export function argmax(q: ArrayLike<number>): Action {
  let best = 0;
  for (let a = 1; a < q.length; a++) if (q[a] > q[best]) best = a;
  return best as Action;
}

export class DqnAgent {
  private online = buildQNetwork();
  private target = buildQNetwork();
  private optimizer: tf.Optimizer;
  private optimizerLr: number;
  /** Pas d'entraînement (décisions) depuis le dernier reset. */
  steps = 0;
  /** Pas qui font décroître ε ; mis à +∞ au chargement d'un modèle entraîné pour démarrer à ε final. */
  private explorationSteps = 0;
  private sinceSync = 0;
  updates = 0;
  /** Perte de Huber de la dernière mise à jour lue (NaN avant la première). */
  lastLoss = Number.NaN;

  constructor(readonly params: DqnParams) {
    this.optimizerLr = params.learningRate;
    this.optimizer = tf.train.adam(this.optimizerLr);
    this.syncTarget();
  }

  get epsilon(): number {
    const { epsStart, epsEnd, epsDecaySteps } = this.params;
    const progress = Math.min(1, this.explorationSteps / epsDecaySteps);
    return epsStart + (epsEnd - epsStart) * progress;
  }

  /** Écrit Q(s, ·) dans `out` (un passage avant du réseau Q). */
  qValues(obs: Float32Array, out: Float32Array): void {
    const q = tf.tidy(() => this.online.predict(tf.tensor2d(obs, [1, OBS_SIZE])) as tf.Tensor);
    out.set(q.dataSync());
    q.dispose();
  }

  /**
   * Choix ε-greedy. Avec probabilité ε l'action est tirée au hasard ; sinon c'est argmax Q.
   * `q` est rempli si `wantQ` est vrai ou si l'action est greedy ; on évite ainsi un passage avant inutile en turbo.
   */
  act(obs: Float32Array, epsilon: number, q: Float32Array, wantQ: boolean): { action: Action; random: boolean; hasQ: boolean } {
    const random = Math.random() < epsilon;
    if (random && !wantQ) return { action: ACTIONS[Math.floor(Math.random() * N_ACTIONS)], random, hasQ: false };
    this.qValues(obs, q);
    const action = random ? ACTIONS[Math.floor(Math.random() * N_ACTIONS)] : argmax(q);
    return { action, random, hasQ: true };
  }

  /** Compte un pas d'interaction : fait avancer ε et, tous les C pas, recopie le réseau Q dans le réseau cible. */
  countStep(): void {
    this.steps++;
    this.explorationSteps++;
    this.sinceSync++;
    if (this.sinceSync >= this.params.targetPeriod) this.syncTarget();
  }

  /** Une descente de gradient sur un mini-batch : Huber(Q(s, a), r + γ·(1 − done)·max Q_cible(s', ·)). */
  learn(batch: Batch): void {
    if (this.params.learningRate !== this.optimizerLr) {
      // Adam garde son pas d'apprentissage en interne : on le recrée (ses moments repartent de zéro).
      this.optimizer.dispose();
      this.optimizerLr = this.params.learningRate;
      this.optimizer = tf.train.adam(this.optimizerLr);
    }
    const { gamma } = this.params;
    const n = batch.size;
    const variables = this.online.trainableWeights.map((w) => w.read() as tf.Variable);
    const cost = tf.tidy(() => {
      const states = tf.tensor2d(batch.states, [n, OBS_SIZE]);
      const nextStates = tf.tensor2d(batch.nextStates, [n, OBS_SIZE]);
      const mask = tf.oneHot(tf.tensor1d(batch.actions, "int32"), N_ACTIONS).toFloat();
      // La cible vient du réseau cible et reste hors du calcul du gradient.
      const maxNext = (this.target.predict(nextStates) as tf.Tensor2D).max(1);
      const targets = tf.tensor1d(batch.rewards).add(maxNext.mul(tf.tensor1d(batch.notDone)).mul(gamma));
      return this.optimizer.minimize(
        () => {
          const q = this.online.apply(states, { training: true }) as tf.Tensor2D;
          const qTaken = q.mul(mask).sum(1);
          return tf.losses.huberLoss(targets, qTaken) as tf.Scalar;
        },
        true,
        variables,
      )!;
    });
    this.updates++;
    if (this.updates % LOSS_EVERY === 0) this.lastLoss = cost.dataSync()[0];
    cost.dispose();
  }

  private syncTarget(): void {
    this.target.setWeights(this.online.getWeights());
    this.sinceSync = 0;
  }

  /** Nouveaux poids aléatoires, nouvel optimiseur, ε remis à sa valeur initiale. */
  reset(): void {
    this.online.dispose();
    this.online = buildQNetwork();
    this.optimizer.dispose();
    this.optimizerLr = this.params.learningRate;
    this.optimizer = tf.train.adam(this.optimizerLr);
    this.steps = 0;
    this.explorationSteps = 0;
    this.updates = 0;
    this.lastLoss = Number.NaN;
    this.syncTarget();
  }

  /** Sauvegarde le réseau Q (ex. « localstorage://tp9-dqn » ou « downloads://model »). */
  async save(destination: string): Promise<void> {
    await this.online.save(destination);
  }

  /** Charge des poids dans le réseau Q et le réseau cible ; ε passe directement à sa valeur finale. */
  async load(source: string | tf.io.IOHandler): Promise<void> {
    const model = await tf.loadLayersModel(source);
    try {
      const input = model.inputs[0].shape;
      const output = model.outputs[0].shape;
      if (input[1] !== OBS_SIZE || output[1] !== N_ACTIONS) {
        throw new Error(`réseau incompatible : entrée ${input[1]}, sortie ${output[1]} (attendu ${OBS_SIZE} → ${N_ACTIONS})`);
      }
      this.online.setWeights(model.getWeights());
    } finally {
      model.dispose();
    }
    this.optimizer.dispose();
    this.optimizer = tf.train.adam(this.optimizerLr);
    this.explorationSteps = Number.POSITIVE_INFINITY;
    this.syncTarget();
  }
}
