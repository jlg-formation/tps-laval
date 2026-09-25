import { loadDecoder, type Decode } from "./decoder";
import { COLORS, SIZE, loadLatentPoints, type Bounds, type Latent } from "./latent";
import { setupPlane } from "./plane";

const GRID = 15;

const $ = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;

const status = $<HTMLParagraphElement>("#status");
const planeCanvas = $<HTMLCanvasElement>("#plane");
const digit = $<HTMLCanvasElement>("#digit");
const coords = $<HTMLOutputElement>("#coords");
const legend = $<HTMLUListElement>("#legend");
const backgroundInputs = document.querySelectorAll<HTMLInputElement>('input[name="background"]');

const format = (v: number) =>
  v.toLocaleString("fr-FR", { minimumFractionDigits: 2, maximumFractionDigits: 2, signDisplay: "exceptZero" });

function toImageData(pixels: Float32Array): ImageData {
  const image = new ImageData(SIZE, SIZE);
  for (let i = 0; i < pixels.length; i++) {
    const v = Math.round(pixels[i] * 255);
    image.data.set([v, v, v, 255], i * 4);
  }
  return image;
}

/** Chiffres décodés au centre de chaque case d'une grille GRID × GRID couvrant le plan. */
async function renderManifold(decode: Decode, bounds: Bounds): Promise<HTMLCanvasElement> {
  const [x0, x1] = bounds.z1;
  const [y0, y1] = bounds.z2;
  const zs: Latent[] = [];
  for (let row = 0; row < GRID; row++) {
    for (let col = 0; col < GRID; col++) {
      zs.push([x0 + ((col + 0.5) / GRID) * (x1 - x0), y1 - ((row + 0.5) / GRID) * (y1 - y0)]);
    }
  }
  const images = await decode(zs);
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = GRID * SIZE;
  const ctx = canvas.getContext("2d")!;
  images.forEach((pixels, i) => {
    ctx.putImageData(toImageData(pixels), (i % GRID) * SIZE, Math.floor(i / GRID) * SIZE);
  });
  return canvas;
}

function renderLegend(): void {
  COLORS.forEach((color, label) => {
    const item = document.createElement("li");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = color;
    item.append(swatch, String(label));
    legend.append(item);
  });
}

async function main(): Promise<void> {
  const [decode, data] = await Promise.all([loadDecoder(), loadLatentPoints()]);

  let scheduled = false;
  let busy = false;
  let pending = false;

  // Une seule inférence à la fois : les déplacements survenus entre-temps sont regroupés.
  function schedule(): void {
    if (busy) {
      pending = true;
    } else if (!scheduled) {
      scheduled = true;
      requestAnimationFrame(() => void update());
    }
  }

  async function update(): Promise<void> {
    scheduled = false;
    busy = true;
    try {
      const z = plane.position;
      const [pixels] = await decode([z]);
      digit.getContext("2d")!.putImageData(toImageData(pixels), 0, 0);
      coords.value = `z = (${format(z[0])} ; ${format(z[1])})`;
    } finally {
      busy = false;
      if (pending) {
        pending = false;
        schedule();
      }
    }
  }

  // Calculée avant d'activer le plan : la session ONNX n'accepte qu'une inférence à la fois.
  const manifold = await renderManifold(decode, data.bounds);
  const plane = setupPlane(planeCanvas, data.bounds, data.points, schedule);

  for (const input of backgroundInputs) {
    input.disabled = false;
    input.addEventListener("change", () => {
      const grid = input.value === "grid";
      plane.setBackground(grid ? manifold : null);
      legend.hidden = grid;
    });
  }

  renderLegend();
  schedule();
  status.textContent = `Modèle chargé : ${data.points.length} points du test (μ de l'encodeur), colorés par chiffre.`;
}

main().catch((error: unknown) => {
  console.error(error);
  status.textContent = "Échec du chargement du modèle. Lancez d'abord « mise run tp6-train ».";
  status.classList.add("error");
});
