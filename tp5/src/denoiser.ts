import { SIZE } from "./noise";

export type Denoiser = (noisy: Float32Array) => Promise<Float32Array>;

export async function loadDenoiser(): Promise<Denoiser> {
  const ort = await import("onnxruntime-web/wasm");
  ort.env.wasm.wasmPaths = new URL("ort/", document.baseURI).href;
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(
    new URL("vae-denoiser.onnx", document.baseURI).href,
  );
  return async (noisy) => {
    const input = new ort.Tensor("float32", noisy, [1, 1, SIZE, SIZE]);
    const results = await session.run({ input });
    return results["output"].data as Float32Array;
  };
}
