// Dessin du jeu sur Canvas 2D : briques, raquette, balle, score, vies et message éventuel.

import {
  BALL_R,
  BRICK_H,
  BRICK_TOP,
  BRICK_W,
  COLS,
  HEIGHT,
  PADDLE_H,
  PADDLE_W,
  PADDLE_Y,
  ROWS,
  WIDTH,
  type Breakout,
} from "./breakout";

const SCALE = 1.2;
const ROW_COLORS = ["#d04848", "#d8803a", "#c8a03a", "#c8c83a", "#48a048", "#4878c8"];
const BACKGROUND = "#11111b";
const TEXT = "#e6e6f0";
const PADDLE = "#7aa2f7";

export class GameRenderer {
  private readonly ctx: CanvasRenderingContext2D;

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(WIDTH * SCALE * dpr);
    canvas.height = Math.round(HEIGHT * SCALE * dpr);
    canvas.style.width = `${WIDTH * SCALE}px`;
    this.ctx.setTransform(SCALE * dpr, 0, 0, SCALE * dpr, 0, 0);
  }

  /** Abscisse dans le repère du jeu d'un événement souris sur le canvas. */
  gameX(event: MouseEvent): number {
    return (event.offsetX / this.canvas.clientWidth) * WIDTH;
  }

  draw(game: Breakout, message?: string): void {
    const ctx = this.ctx;
    ctx.fillStyle = BACKGROUND;
    ctx.fillRect(0, 0, WIDTH, HEIGHT);

    for (let r = 0; r < ROWS; r++) {
      ctx.fillStyle = ROW_COLORS[r];
      for (let c = 0; c < COLS; c++) {
        if (game.bricks[r * COLS + c]) ctx.fillRect(c * BRICK_W + 1, BRICK_TOP + r * BRICK_H + 1, BRICK_W - 2, BRICK_H - 2);
      }
    }

    ctx.fillStyle = PADDLE;
    ctx.fillRect(game.paddleX - PADDLE_W / 2, PADDLE_Y, PADDLE_W, PADDLE_H);

    ctx.fillStyle = TEXT;
    ctx.beginPath();
    ctx.arc(game.ballX, game.ballY, BALL_R, 0, 2 * Math.PI);
    ctx.fill();

    ctx.font = "600 16px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    ctx.fillText(`Score ${game.score}`, 12, 26);
    for (let i = 0; i < game.lives; i++) {
      ctx.beginPath();
      ctx.arc(WIDTH - 16 - i * 16, 26, 5, 0, 2 * Math.PI);
      ctx.fill();
    }

    if (message) {
      ctx.fillStyle = "rgba(17, 17, 27, 0.8)";
      ctx.fillRect(0, HEIGHT / 2 - 34, WIDTH, 68);
      ctx.fillStyle = TEXT;
      ctx.textAlign = "center";
      const lines = message.split("\n");
      lines.forEach((line, i) => ctx.fillText(line, WIDTH / 2, HEIGHT / 2 + (i - (lines.length - 1) / 2) * 22));
    }
  }
}
