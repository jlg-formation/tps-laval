import { SIZE, type Latent } from "./latent";

/** Décode plusieurs codes latents en une seule inférence ; une image 28×28 dans [0, 1] par code. */
export type Decode = (zs: readonly Latent[]) => Promise<Float32Array[]>;

export async function loadDecoder(): Promise<Decode> {
  const ort = await import("onnxruntime-web/wasm");
  ort.env.wasm.wasmPaths = new URL("ort/", document.baseURI).href;
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(new URL("decoder.onnx", document.baseURI).href);
  return async (zs) => {
    const latent = new ort.Tensor("float32", Float32Array.from(zs.flat()), [zs.length, 2]);
    const results = await session.run({ latent });
    const data = results["output"].data as Float32Array;
    return zs.map((_, i) => data.subarray(i * SIZE * SIZE, (i + 1) * SIZE * SIZE));
  };
}
