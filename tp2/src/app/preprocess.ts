const SIZE = 28;
const BOX = 20;
const INK_THRESHOLD = 0.05;

/**
 * Convertit le dessin (trait noir sur fond blanc) au format MNIST :
 * 28×28, trait blanc sur fond noir, chiffre tenu dans 20×20 et centré par centre de masse.
 * Retourne null si le canvas est vide.
 */
export function preprocess(source: HTMLCanvasElement): Float32Array<ArrayBuffer> | null {
  const { width, height } = source;
  const data = source.getContext('2d', { willReadFrequently: true })!.getImageData(0, 0, width, height).data;

  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (1 - data[(y * width + x) * 4] / 255 > INK_THRESHOLD) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }
    }
  }
  if (maxX < 0) {
    return null;
  }

  const boxW = maxX - minX + 1;
  const boxH = maxY - minY + 1;
  const scale = BOX / Math.max(boxW, boxH);
  const w = Math.max(1, Math.round(boxW * scale));
  const h = Math.max(1, Math.round(boxH * scale));

  const small = document.createElement('canvas');
  small.width = w;
  small.height = h;
  const ctx = small.getContext('2d', { willReadFrequently: true })!;
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, w, h);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, minX, minY, boxW, boxH, 0, 0, w, h);
  const smallData = ctx.getImageData(0, 0, w, h).data;

  const ink = new Float32Array(w * h);
  let mass = 0;
  let cx = 0;
  let cy = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = 1 - smallData[(y * w + x) * 4] / 255;
      ink[y * w + x] = v;
      mass += v;
      cx += x * v;
      cy += y * v;
    }
  }
  cx /= mass || 1;
  cy /= mass || 1;

  const offsetX = clamp(Math.round(SIZE / 2 - cx), 0, SIZE - w);
  const offsetY = clamp(Math.round(SIZE / 2 - cy), 0, SIZE - h);

  const out = new Float32Array(SIZE * SIZE);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      out[(y + offsetY) * SIZE + x + offsetX] = ink[y * w + x];
    }
  }
  return out;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}
