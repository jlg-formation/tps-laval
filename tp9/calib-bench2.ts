import * as tf from "@tensorflow/tfjs";

await tf.setBackend("cpu");
const time = (label: string, n: number, f: () => void) => {
  f();
  const t = performance.now();
  for (let i = 0; i < n; i++) f();
  console.log(label, ((performance.now() - t) / n).toFixed(2), "ms");
};
const a = tf.randomNormal([64, 128]);
const b = tf.randomNormal([128, 128]);
time("matMul 64x128x128", 50, () => tf.tidy(() => tf.matMul(a, b).dataSync()));
const x = tf.randomNormal([64, 65]);
const w1 = tf.variable(tf.randomNormal([65, 128]));
const w2 = tf.variable(tf.randomNormal([128, 128]));
const w3 = tf.variable(tf.randomNormal([128, 3]));
const opt = tf.train.adam(1e-3);
time("raw minimize", 20, () =>
  tf.tidy(() => {
    opt.minimize(() => x.matMul(w1).relu().matMul(w2).relu().matMul(w3).square().mean() as tf.Scalar, true)!.dataSync();
  }),
);
time("raw grads only", 20, () =>
  tf.tidy(() => {
    const { value, grads } = tf.variableGrads(() => x.matMul(w1).relu().matMul(w2).relu().matMul(w3).square().mean() as tf.Scalar);
    value.dataSync();
    Object.values(grads).forEach((g) => g.dataSync());
  }),
);
