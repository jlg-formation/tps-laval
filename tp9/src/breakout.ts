// Environnement Breakout : physique à pas fixe, règles, observation et récompenses. Aucun lien avec le rendu.

export const WIDTH = 400;
export const HEIGHT = 460;

export const ROWS = 6;
export const COLS = 10;
export const BRICK_W = WIDTH / COLS;
export const BRICK_H = 16;
export const BRICK_TOP = 60;

export const PADDLE_W = 64;
export const PADDLE_H = 10;
export const PADDLE_Y = HEIGHT - 36;
export const PADDLE_SPEED = 7;

export const BALL_R = 4;
const BASE_SPEED = 4;
const SPEED_STEP = 1;
// Plafond : à 3 sous-pas, la balle avance d'au plus 2,5 unités, moins que son rayon et que l'épaisseur des objets.
export const MAX_SPEED = 7.5;
const SUBSTEPS = 3;
// Renvois de raquette après lesquels la balle accélère (comme le jeu d'origine).
const SPEEDUP_HITS = [4, 12];
// Rangées du haut (0 = tout en haut) dont le premier contact accélère la balle.
const TOP_ROWS = 2;

const MAX_BOUNCE = (60 * Math.PI) / 180;
const MAX_LAUNCH = (45 * Math.PI) / 180;

export const LIVES = 3;
export const MAX_STEPS = 10_000;

export type Action = 0 | 1 | 2;
export const ACTIONS: readonly Action[] = [0, 1, 2];
export const ACTION_NAMES = ["gauche", "rester", "droite"] as const;

/** Balle (x, y, vx, vy) + raquette (x) + 60 briques. */
export const OBS_SIZE = 5 + ROWS * COLS;

export interface StepResult {
  reward: number;
  lifeLost: boolean;
  terminal: boolean;
  truncated: boolean;
}

export class Breakout {
  readonly bricks = new Uint8Array(ROWS * COLS);
  bricksLeft = 0;
  paddleX = WIDTH / 2;
  ballX = 0;
  ballY = 0;
  vx = 0;
  vy = 0;
  lives = LIVES;
  score = 0;
  steps = 0;
  private speedLevel = 0;
  private paddleHits = 0;
  private reachedTop = false;

  constructor(private readonly random: () => number = Math.random) {
    this.reset();
  }

  reset(): void {
    this.bricks.fill(1);
    this.bricksLeft = this.bricks.length;
    this.paddleX = WIDTH / 2;
    this.lives = LIVES;
    this.score = 0;
    this.steps = 0;
    this.launch();
  }

  get speed(): number {
    return Math.min(MAX_SPEED, BASE_SPEED + this.speedLevel * SPEED_STEP);
  }

  /** Nouvelle balle posée sur la raquette, lancée vers le haut avec un angle aléatoire : seul hasard du jeu. */
  private launch(): void {
    this.speedLevel = 0;
    this.paddleHits = 0;
    this.reachedTop = false;
    const angle = (this.random() * 2 - 1) * MAX_LAUNCH;
    this.ballX = this.paddleX;
    this.ballY = PADDLE_Y - BALL_R - 1;
    this.vx = this.speed * Math.sin(angle);
    this.vy = -this.speed * Math.cos(angle);
  }

  private speedUp(): void {
    this.speedLevel++;
    const k = this.speed / Math.hypot(this.vx, this.vy);
    this.vx *= k;
    this.vy *= k;
  }

  step(action: Action): StepResult {
    this.steps++;
    const half = PADDLE_W / 2;
    this.paddleX = Math.min(WIDTH - half, Math.max(half, this.paddleX + (action - 1) * PADDLE_SPEED));

    let reward = 0;
    let lifeLost = false;
    for (let i = 0; i < SUBSTEPS && !lifeLost; i++) {
      this.ballX += this.vx / SUBSTEPS;
      this.ballY += this.vy / SUBSTEPS;
      this.collideWalls();
      this.collidePaddle();
      reward += this.collideBricks();
      lifeLost = this.ballY - BALL_R > HEIGHT;
    }

    if (lifeLost) {
      reward -= 1;
      this.lives--;
      if (this.lives > 0) this.launch();
    }
    const terminal = this.lives === 0 || this.bricksLeft === 0;
    return { reward, lifeLost, terminal, truncated: !terminal && this.steps >= MAX_STEPS };
  }

  private collideWalls(): void {
    if (this.ballX - BALL_R < 0) {
      this.ballX = BALL_R;
      this.vx = Math.abs(this.vx);
    } else if (this.ballX + BALL_R > WIDTH) {
      this.ballX = WIDTH - BALL_R;
      this.vx = -Math.abs(this.vx);
    }
    if (this.ballY - BALL_R < 0) {
      this.ballY = BALL_R;
      this.vy = Math.abs(this.vy);
    }
  }

  private collidePaddle(): void {
    if (this.vy <= 0) return;
    if (this.ballY + BALL_R < PADDLE_Y || this.ballY - BALL_R > PADDLE_Y + PADDLE_H) return;
    const reach = PADDLE_W / 2 + BALL_R;
    const offset = (this.ballX - this.paddleX) / reach;
    if (Math.abs(offset) > 1) return;
    // Centre : renvoi vertical ; bords : renvoi à plat (jusqu'à 60° de la verticale).
    const angle = offset * MAX_BOUNCE;
    this.ballY = PADDLE_Y - BALL_R;
    this.paddleHits++;
    if (SPEEDUP_HITS.includes(this.paddleHits)) this.speedLevel++;
    this.vx = this.speed * Math.sin(angle);
    this.vy = -this.speed * Math.cos(angle);
  }

  /** Casse au plus une brique par sous-pas et renvoie la récompense (+1 par brique). */
  private collideBricks(): number {
    const left = this.ballX - BALL_R;
    const right = this.ballX + BALL_R;
    const top = this.ballY - BALL_R;
    const bottom = this.ballY + BALL_R;
    const c0 = Math.max(0, Math.floor(left / BRICK_W));
    const c1 = Math.min(COLS - 1, Math.floor(right / BRICK_W));
    const r0 = Math.max(0, Math.floor((top - BRICK_TOP) / BRICK_H));
    const r1 = Math.min(ROWS - 1, Math.floor((bottom - BRICK_TOP) / BRICK_H));

    let hit = -1;
    let hitX = 0;
    let hitY = 0;
    let bestArea = 0;
    for (let r = r0; r <= r1; r++) {
      for (let c = c0; c <= c1; c++) {
        if (!this.bricks[r * COLS + c]) continue;
        const bx = c * BRICK_W;
        const by = BRICK_TOP + r * BRICK_H;
        const ox = Math.min(right, bx + BRICK_W) - Math.max(left, bx);
        const oy = Math.min(bottom, by + BRICK_H) - Math.max(top, by);
        if (ox > 0 && oy > 0 && ox * oy > bestArea) {
          bestArea = ox * oy;
          hit = r * COLS + c;
          hitX = ox;
          hitY = oy;
        }
      }
    }
    if (hit < 0) return 0;

    const row = Math.floor(hit / COLS);
    const cx = (hit % COLS) * BRICK_W + BRICK_W / 2;
    const cy = BRICK_TOP + row * BRICK_H + BRICK_H / 2;
    // La balle rebondit selon l'axe où elle a le moins pénétré la brique.
    if (hitX < hitY) this.vx = this.ballX < cx ? -Math.abs(this.vx) : Math.abs(this.vx);
    else this.vy = this.ballY < cy ? -Math.abs(this.vy) : Math.abs(this.vy);

    this.bricks[hit] = 0;
    this.bricksLeft--;
    this.score++;
    if (row < TOP_ROWS && !this.reachedTop) {
      this.reachedTop = true;
      this.speedUp();
    }
    return 1;
  }

  /** Observation normalisée : positions dans [0, 1], vitesses dans [−1, 1], briques 0/1. */
  observe(out: Float32Array): Float32Array {
    out[0] = this.ballX / WIDTH;
    out[1] = this.ballY / HEIGHT;
    out[2] = this.vx / MAX_SPEED;
    out[3] = this.vy / MAX_SPEED;
    out[4] = this.paddleX / WIDTH;
    out.set(this.bricks, 5);
    return out;
  }
}
