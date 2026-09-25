export const SIZE = 28;

/** Génère plusieurs images en une seule inférence ; une image 28×28 dans [0, 1] par bruit z. */
export type Generate = (zs: readonly Float32Array[]) => Promise<Float32Array[]>;

export interface Generator {
  /** Dimension du bruit z, lue dans le modèle : elle suit l'option --latent de train.py. */
  latent: number;
  generate: Generate;
}

export async function loadGenerator(): Promise<Generator> {
  const ort = await import("onnxruntime-web/wasm");
  ort.env.wasm.wasmPaths = new URL("ort/", document.baseURI).href;
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(new URL("generator.onnx", document.baseURI).href);
  const [input] = session.inputMetadata;
  if (!input.isTensor || typeof input.shape[1] !== "number") {
    throw new Error("generator.onnx : entrée « latent » de forme [batch, latent] attendue.");
  }
  const latent = input.shape[1];
  const generate: Generate = async (zs) => {
    const data = new Float32Array(zs.length * latent);
    zs.forEach((z, i) => data.set(z, i * latent));
    const results = await session.run({ latent: new ort.Tensor("float32", data, [zs.length, latent]) });
    // Sortie tanh du générateur : [-1, 1] -> [0, 1].
    const pixels = (results["output"].data as Float32Array).map((v) => (v + 1) / 2);
    return zs.map((_, i) => pixels.subarray(i * SIZE * SIZE, (i + 1) * SIZE * SIZE));
  };
  return { latent, generate };
}
