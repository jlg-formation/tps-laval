import { SIZE, loadGenerator, type Generator } from "./generator";

const GRID = 8;

const $ = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;

const status = $<HTMLParagraphElement>("#status");
const grid = $<HTMLCanvasElement>("#grid");
const button = $<HTMLButtonElement>("#generate");

/** Tirage selon N(0, 1) par la méthode de Box-Muller. */
function randn(): number {
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}


function toImageData(pixels: Float32Array): ImageData {
  const image = new ImageData(SIZE, SIZE);
  for (let i = 0; i < pixels.length; i++) {
    const v = Math.round(pixels[i] * 255);
    image.data.set([v, v, v, 255], i * 4);
  }
  return image;
}

async function render({ latent, generate }: Generator): Promise<void> {
  const zs = Array.from({ length: GRID * GRID }, () => Float32Array.from({ length: latent }, randn));
  const images = await generate(zs);
  const ctx = grid.getContext("2d")!;
  images.forEach((pixels, i) => {
    ctx.putImageData(toImageData(pixels), (i % GRID) * SIZE, Math.floor(i / GRID) * SIZE);
  });
}

async function main(): Promise<void> {
  const generator = await loadGenerator();
  await render(generator);
  status.textContent = `Modèle chargé : ${GRID * GRID} chiffres générés à partir de bruits z de dimension ${generator.latent}.`;

  button.disabled = false;
  button.addEventListener("click", async () => {
    // La session ONNX n'accepte qu'une inférence à la fois.
    button.disabled = true;
    try {
      await render(generator);
    } finally {
      button.disabled = false;
    }
  });
}

main().catch((error: unknown) => {
  console.error(error);
  status.textContent = "Échec du chargement du modèle. Lancez d'abord « mise run tp7-train ».";
  status.classList.add("error");
});
