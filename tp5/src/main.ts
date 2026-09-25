import { loadDenoiser } from "./denoiser";
import { setupDrawingPad } from "./drawing-pad";
import {
  NOISE_MAX,
  NOISE_STEP,
  NO_NOISE,
  SIZE,
  applyNoise,
  newRealization,
  type NoiseLevels,
} from "./noise";
import { preprocess } from "./preprocess";

const $ = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;

const status = $<HTMLParagraphElement>("#status");
const padCanvas = $<HTMLCanvasElement>("#pad");
const previews = {
  clean: $<HTMLCanvasElement>("#clean"),
  noisy: $<HTMLCanvasElement>("#noisy"),
  denoised: $<HTMLCanvasElement>("#denoised"),
};
const controls: Record<keyof NoiseLevels, { button: HTMLButtonElement; label: HTMLOutputElement }> = {
  sigma: { button: $("#add-gauss"), label: $("#level-gauss") },
  saltPepper: { button: $("#add-salt-pepper"), label: $("#level-salt-pepper") },
  blocks: { button: $("#add-blocks"), label: $("#level-blocks") },
};
const removeNoise = $<HTMLButtonElement>("#remove-noise");
const clear = $<HTMLButtonElement>("#clear");

const formatters: Record<keyof NoiseLevels, (value: number) => string> = {
  sigma: (v) => `σ = ${v.toLocaleString("fr-FR", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
  saltPepper: (v) => `${(v * 100).toLocaleString("fr-FR", { maximumFractionDigits: 1 })} %`,
  blocks: (v) => `${v} bloc${v > 1 ? "s" : ""}`,
};

function render(canvas: HTMLCanvasElement, pixels: Float32Array | null): void {
  const image = new ImageData(SIZE, SIZE);
  if (pixels) {
    for (let i = 0; i < pixels.length; i++) {
      const v = Math.round(pixels[i] * 255);
      image.data.set([v, v, v, 255], i * 4);
    }
  } else {
    for (let i = 3; i < image.data.length; i += 4) {
      image.data[i] = 255;
    }
  }
  canvas.getContext("2d")!.putImageData(image, 0, 0);
}

async function main(): Promise<void> {
  const denoise = await loadDenoiser();

  let levels: NoiseLevels = { ...NO_NOISE };
  let realization = newRealization();
  let scheduled = false;
  let busy = false;
  let pending = false;

  // Une seule inférence à la fois : les changements survenus entre-temps sont regroupés.
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
      const clean = preprocess(padCanvas);
      const noisy = clean && applyNoise(clean, levels, realization);
      render(previews.clean, clean);
      render(previews.noisy, noisy);
      render(previews.denoised, noisy && (await denoise(noisy)));
    } finally {
      busy = false;
      if (pending) {
        pending = false;
        schedule();
      }
    }
  }

  function refreshControls(): void {
    for (const key of Object.keys(controls) as (keyof NoiseLevels)[]) {
      controls[key].label.value = formatters[key](levels[key]);
      controls[key].button.disabled = levels[key] >= NOISE_MAX[key];
    }
    removeNoise.disabled = Object.values(levels).every((v) => v === 0);
  }

  const pad = setupDrawingPad(padCanvas, schedule);

  for (const key of Object.keys(controls) as (keyof NoiseLevels)[]) {
    controls[key].button.addEventListener("click", () => {
      levels = { ...levels, [key]: Math.min(NOISE_MAX[key], levels[key] + NOISE_STEP[key]) };
      realization = newRealization();
      refreshControls();
      schedule();
    });
  }
  removeNoise.addEventListener("click", () => {
    levels = { ...NO_NOISE };
    refreshControls();
    schedule();
  });
  clear.addEventListener("click", () => {
    realization = newRealization();
    pad.clear();
  });

  clear.disabled = false;
  refreshControls();
  status.textContent = "Modèle chargé. Dessinez un chiffre, puis ajoutez du bruit.";
}

main().catch((error: unknown) => {
  console.error(error);
  status.textContent = "Échec du chargement du modèle. Lancez d'abord « mise run tp5-train ».";
  status.classList.add("error");
});
