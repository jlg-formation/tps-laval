import { Service, signal } from '@angular/core';

export type ClassifierStatus = 'loading' | 'ready' | 'error';

@Service()
export class DigitClassifier {
  private readonly statusState = signal<ClassifierStatus>('loading');
  readonly status = this.statusState.asReadonly();

  private readonly runtime = this.load();

  async predict(pixels: Float32Array<ArrayBuffer>): Promise<number[]> {
    const runtime = await this.runtime;
    if (!runtime) {
      throw new Error('Modèle ONNX indisponible');
    }
    const { ort, session } = runtime;
    const results = await session.run({ input: new ort.Tensor('float32', pixels, [1, 1, 28, 28]) });
    return softmax(Array.from(results['output'].data as Float32Array));
  }

  private async load() {
    try {
      // Import dynamique : le runtime ONNX reste hors du bundle initial.
      const ort = await import('onnxruntime-web/wasm');
      ort.env.wasm.wasmPaths = new URL('ort/', document.baseURI).href;
      ort.env.wasm.numThreads = 1;
      const session = await ort.InferenceSession.create(new URL('mlp_mnist.onnx', document.baseURI).href);
      this.statusState.set('ready');
      return { ort, session };
    } catch (error) {
      console.error('Échec du chargement du modèle ONNX', error);
      this.statusState.set('error');
      return null;
    }
  }
}

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}
