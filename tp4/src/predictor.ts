export interface Metric {
  name: string;
  mae: number;
  rmse: number;
}

/** Contenu de ticket-rnn.json, écrit par tp4/train.py. */
export interface ModelInfo {
  cell: string;
  window: number;
  hidden: number;
  metrics: Metric[];
  test: { dates: string[]; actual: number[]; predicted: number[]; lastWeek: number[] };
  lastWindow: { dates: string[]; values: number[] };
  nextPrediction: number;
}

export type Predictor = (tickets: number[]) => Promise<number>;

export async function loadInfo(): Promise<ModelInfo> {
  const response = await fetch(new URL("ticket-rnn.json", document.baseURI));
  if (!response.ok) {
    throw new Error(`ticket-rnn.json introuvable (${response.status})`);
  }
  return response.json();
}

export async function loadPredictor(): Promise<Predictor> {
  const ort = await import("onnxruntime-web/wasm");
  ort.env.wasm.wasmPaths = new URL("ort/", document.baseURI).href;
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(
    new URL("ticket-rnn.onnx", document.baseURI).href,
  );
  return async (tickets) => {
    const input = new ort.Tensor("float32", Float32Array.from(tickets), [1, tickets.length, 1]);
    const results = await session.run({ input });
    return (results["output"].data as Float32Array)[0];
  };
}
