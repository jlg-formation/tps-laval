// Plages de bruit : doivent rester identiques à celles de train.py.
export const SIZE = 28;
const PIXELS = SIZE * SIZE;
const BLOCK_MIN = 6;
const BLOCK_MAX = 10;

export interface NoiseLevels {
  sigma: number;
  saltPepper: number;
  blocks: number;
}

export const NOISE_STEP: NoiseLevels = { sigma: 0.15, saltPepper: 0.075, blocks: 1 };
export const NOISE_MAX: NoiseLevels = { sigma: 0.6, saltPepper: 0.3, blocks: 3 };
export const NO_NOISE: NoiseLevels = { sigma: 0, saltPepper: 0, blocks: 0 };

interface Block {
  x: number;
  y: number;
  side: number;
}

/** Tirage aléatoire figé : seule l'intensité change tant qu'il n'est pas renouvelé. */
export interface NoiseRealization {
  gauss: Float32Array;
  hit: Float32Array;
  salt: Uint8Array;
  blocks: Block[];
}

export function newRealization(): NoiseRealization {
  const gauss = new Float32Array(PIXELS);
  for (let i = 0; i < PIXELS; i++) {
    // Box-Muller : N(0, 1) à partir de deux uniformes.
    gauss[i] = Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
  }
  const hit = Float32Array.from({ length: PIXELS }, () => Math.random());
  const salt = Uint8Array.from({ length: PIXELS }, () => (Math.random() < 0.5 ? 1 : 0));
  const blocks = Array.from({ length: NOISE_MAX.blocks }, () => {
    const side = BLOCK_MIN + Math.floor(Math.random() * (BLOCK_MAX - BLOCK_MIN + 1));
    return {
      x: Math.floor(Math.random() * (SIZE - side + 1)),
      y: Math.floor(Math.random() * (SIZE - side + 1)),
      side,
    };
  });
  return { gauss, hit, salt, blocks };
}

/** Gaussien, puis poivre et sel, puis occlusion : même ordre qu'à l'entraînement. */
export function applyNoise(clean: Float32Array, levels: NoiseLevels, r: NoiseRealization): Float32Array {
  const out = new Float32Array(PIXELS);
  for (let i = 0; i < PIXELS; i++) {
    const v = Math.min(1, Math.max(0, clean[i] + levels.sigma * r.gauss[i]));
    out[i] = r.hit[i] < levels.saltPepper ? r.salt[i] : v;
  }
  for (const { x, y, side } of r.blocks.slice(0, levels.blocks)) {
    for (let row = y; row < y + side; row++) {
      out.fill(0, row * SIZE + x, row * SIZE + x + side);
    }
  }
  return out;
}
