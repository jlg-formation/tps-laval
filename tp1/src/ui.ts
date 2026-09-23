// ---------------------------------------------------------------------------
// Câblage de l'interface : récupère les éléments du DOM, branche les
// événements et met à jour les affichages temps réel.
// ---------------------------------------------------------------------------

import type { Perceptron } from "./perceptron";
import { state } from "./state";

/** Callbacks fournis par main.ts pour réagir aux actions utilisateur. */
export interface UiHandlers {
  onToggleClass: () => void;
  onDemo: () => void;
  onReset: () => void;
  onToggleTraining: () => void;
  onStep: () => void;
}

function byId<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Élément #${id} introuvable`);
  return el as T;
}

export interface Ui {
  refreshTraining(): void;
  refreshActiveClass(): void;
  updateReadouts(perceptron: Perceptron, errors: number): void;
}

export function setupUi(handlers: UiHandlers): Ui {
  const toggleClassBtn = byId<HTMLButtonElement>("toggle-class");
  const activeClassLabel = byId<HTMLElement>("active-class");
  const demoBtn = byId<HTMLButtonElement>("demo");
  const resetBtn = byId<HTMLButtonElement>("reset");
  const trainBtn = byId<HTMLButtonElement>("train");
  const stepBtn = byId<HTMLButtonElement>("step");
  const speedInput = byId<HTMLInputElement>("speed");
  const speedVal = byId<HTMLElement>("speed-val");
  const lrInput = byId<HTMLInputElement>("lr");
  const lrVal = byId<HTMLElement>("lr-val");

  const w1El = byId<HTMLElement>("w1");
  const w2El = byId<HTMLElement>("w2");
  const bEl = byId<HTMLElement>("b");
  const iterEl = byId<HTMLElement>("iter");
  const errEl = byId<HTMLElement>("err");

  toggleClassBtn.addEventListener("click", handlers.onToggleClass);
  demoBtn.addEventListener("click", handlers.onDemo);
  resetBtn.addEventListener("click", handlers.onReset);
  trainBtn.addEventListener("click", handlers.onToggleTraining);
  stepBtn.addEventListener("click", handlers.onStep);

  speedInput.addEventListener("input", () => {
    state.speed = Number(speedInput.value);
    speedVal.textContent = speedInput.value;
  });

  lrInput.addEventListener("input", () => {
    state.learningRate = Number(lrInput.value);
    lrVal.textContent = state.learningRate.toFixed(2);
  });

  // Initialise les affichages à partir des valeurs par défaut de l'état.
  speedInput.value = String(state.speed);
  speedVal.textContent = String(state.speed);
  lrInput.value = String(state.learningRate);
  lrVal.textContent = state.learningRate.toFixed(2);

  const ui: Ui = {
    refreshTraining() {
      trainBtn.textContent = state.training ? "Pause" : "Entraîner";
    },
    refreshActiveClass() {
      activeClassLabel.textContent = state.activeLabel === 1 ? "Noir" : "Blanc";
    },
    updateReadouts(perceptron, errors) {
      w1El.textContent = perceptron.w1.toFixed(3);
      w2El.textContent = perceptron.w2.toFixed(3);
      bEl.textContent = perceptron.b.toFixed(3);
      iterEl.textContent = String(perceptron.iterations);
      errEl.textContent = String(errors);
    },
  };

  ui.refreshTraining();
  ui.refreshActiveClass();
  return ui;
}
