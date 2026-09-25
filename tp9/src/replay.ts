// Replay buffer circulaire : on stocke les transitions (s, a, r, s', done) et on en tire des mini-batchs uniformes.

export interface Batch {
  size: number;
  states: Float32Array;
  actions: Int32Array;
  rewards: Float32Array;
  nextStates: Float32Array;
  /** 1 − done : multiplie γ·max Q(s', ·), coupé à 0 pour un état terminal. */
  notDone: Float32Array;
}

export class ReplayBuffer {
  private readonly states: Float32Array;
  private readonly nextStates: Float32Array;
  private readonly actions: Int32Array;
  private readonly rewards: Float32Array;
  private readonly dones: Uint8Array;
  private readonly batch: Batch;
  private next = 0;
  size = 0;

  constructor(
    readonly capacity: number,
    private readonly obsSize: number,
    batchSize: number,
  ) {
    this.states = new Float32Array(capacity * obsSize);
    this.nextStates = new Float32Array(capacity * obsSize);
    this.actions = new Int32Array(capacity);
    this.rewards = new Float32Array(capacity);
    this.dones = new Uint8Array(capacity);
    this.batch = {
      size: batchSize,
      states: new Float32Array(batchSize * obsSize),
      actions: new Int32Array(batchSize),
      rewards: new Float32Array(batchSize),
      nextStates: new Float32Array(batchSize * obsSize),
      notDone: new Float32Array(batchSize),
    };
  }

  push(s: Float32Array, a: number, r: number, next: Float32Array, done: boolean): void {
    const i = this.next;
    this.states.set(s, i * this.obsSize);
    this.nextStates.set(next, i * this.obsSize);
    this.actions[i] = a;
    this.rewards[i] = r;
    this.dones[i] = done ? 1 : 0;
    // Buffer plein : la plus ancienne transition est écrasée.
    this.next = (i + 1) % this.capacity;
    this.size = Math.min(this.size + 1, this.capacity);
  }

  /** Tire un mini-batch avec remise ; les tableaux renvoyés sont réutilisés d'un appel à l'autre. */
  sample(): Batch {
    const b = this.batch;
    const n = this.obsSize;
    for (let k = 0; k < b.size; k++) {
      const i = Math.floor(Math.random() * this.size);
      b.states.set(this.states.subarray(i * n, (i + 1) * n), k * n);
      b.nextStates.set(this.nextStates.subarray(i * n, (i + 1) * n), k * n);
      b.actions[k] = this.actions[i];
      b.rewards[k] = this.rewards[i];
      b.notDone[k] = 1 - this.dones[i];
    }
    return b;
  }

  clear(): void {
    this.next = 0;
    this.size = 0;
  }
}
