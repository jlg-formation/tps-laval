import * as tf from "@tensorflow/tfjs";
import { DqnAgent, type DqnParams } from "./src/dqn";
import { Trainer } from "./src/training";

const params: DqnParams = {
  learningRate: Number(process.env.LR ?? 1e-3),
  gamma: Number(process.env.GAMMA ?? 0.99),
  epsStart: 1,
  epsEnd: 0.05,
  epsDecaySteps: Number(process.env.DECAY ?? 50_000),
  targetPeriod: Number(process.env.C ?? 1_000),
};
const maxDecisions = Number(process.env.STEPS ?? 60_000);

await tf.setBackend("cpu");
const agent = new DqnAgent(params);
const trainer = new Trainer(agent);
trainer.wantQ = false;
const t0 = performance.now();
let printed = 0;
while (agent.steps < maxDecisions) {
  trainer.tick();
  const h = trainer.history;
  if (h.length >= printed + 20) {
    printed = h.length;
    const last = h.slice(-20).map((e) => e.score);
    const mean = last.reduce((a, b) => a + b) / last.length;
    const s = (performance.now() - t0) / 1000;
    console.log(
      `ep ${h.length} steps ${agent.steps} eps ${agent.epsilon.toFixed(2)} mean20 ${mean.toFixed(1)} loss ${agent.lastLoss.toExponential(2)} ${s.toFixed(0)} s (${(agent.steps / s).toFixed(0)} dec/s) tensors ${tf.memory().numTensors}`,
    );
  }
}
if (process.env.SAVE) {
  const { writeFileSync, mkdirSync } = await import("node:fs");
  await (agent as unknown as { online: tf.LayersModel }).online.save(
    tf.io.withSaveHandler(async (artifacts) => {
      mkdirSync(process.env.SAVE!, { recursive: true });
      const { weightData, ...rest } = artifacts;
      const json = { ...rest, weightsManifest: [{ paths: ["model.weights.bin"], weights: artifacts.weightSpecs }] };
      delete (json as Record<string, unknown>).weightSpecs;
      writeFileSync(`${process.env.SAVE}/model.json`, JSON.stringify(json));
      writeFileSync(`${process.env.SAVE}/model.weights.bin`, Buffer.from(weightData as ArrayBuffer));
      return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: "JSON" } };
    }),
  );
  console.log("saved");
}
