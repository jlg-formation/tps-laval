import { Breakout, PADDLE_SPEED, type Action } from "./src/breakout";
import { randomBaseline } from "./src/training";

const t0 = performance.now();
console.log("baseline", randomBaseline(200), `${(performance.now() - t0).toFixed(0)} ms`);

// Politique heuristique : la raquette suit la balle.
const env = new Breakout();
const scores: number[] = [];
const lengths: number[] = [];
let truncated = 0;
for (let g = 0; g < 50; g++) {
  env.reset();
  for (;;) {
    const d = env.ballX - env.paddleX;
    const a: Action = Math.abs(d) < PADDLE_SPEED / 2 ? 1 : d < 0 ? 0 : 2;
    const r = env.step(a);
    if (r.terminal || r.truncated) {
      if (r.truncated) truncated++;
      break;
    }
  }
  scores.push(env.score);
  lengths.push(env.steps);
}
console.log("tracker mean", scores.reduce((a, b) => a + b) / scores.length, "truncated", truncated);
console.log(scores.join(" "));
console.log("steps", lengths.join(" "));
