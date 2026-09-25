import * as tf from "@tensorflow/tfjs";
import { Breakout, OBS_SIZE } from "./src/breakout";
import { DqnAgent } from "./src/dqn";
import { ReplayBuffer } from "./src/replay";

await tf.setBackend("cpu");
const agent = new DqnAgent({ learningRate: 1e-3, gamma: 0.99, epsStart: 1, epsEnd: 0.05, epsDecaySteps: 50_000, targetPeriod: 1_000 });
const env = new Breakout();
const buf = new ReplayBuffer(5000, OBS_SIZE, 64);
const s = new Float32Array(OBS_SIZE);
const s2 = new Float32Array(OBS_SIZE);
for (let i = 0; i < 2000; i++) {
  env.observe(s);
  const r = env.step(1);
  env.observe(s2);
  buf.push(s, 1, r.reward, s2, r.lifeLost);
  if (r.terminal) env.reset();
}
const q = new Float32Array(3);
const time = (label: string, n: number, f: () => void) => {
  f();
  const t = performance.now();
  for (let i = 0; i < n; i++) f();
  console.log(label, ((performance.now() - t) / n).toFixed(2), "ms");
};
time("qValues", 200, () => agent.qValues(s, q));
time("sample", 200, () => buf.sample());
time("learn", 50, () => agent.learn(buf.sample()));
time("sync", 50, () => agent.countStep());
