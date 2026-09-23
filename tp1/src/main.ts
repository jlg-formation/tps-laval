// ---------------------------------------------------------------------------
// Point d'entrée : relie l'état, le perceptron, le rendu et l'interface.
// ---------------------------------------------------------------------------

import { toNormalized } from "./geometry";
import { Perceptron, type TrainingSample } from "./perceptron";
import { draw, pointRadius } from "./renderer";
import { makeDemoPoints, state } from "./state";
import { setupUi } from "./ui";

const canvas = document.getElementById("board") as HTMLCanvasElement;
const ctx = canvas.getContext("2d");
if (!ctx) throw new Error("Contexte 2D indisponible");

const W = canvas.width;
const H = canvas.height;

const perceptron = new Perceptron();

/** Convertit les points (pixels) en exemples d'entraînement (normalisés). */
function samples(): TrainingSample[] {
  return state.points.map((p) => {
    const n = toNormalized(p.px, p.py, W, H);
    return { x: n.x, y: n.y, label: p.label };
  });
}

const ui = setupUi({
  onToggleClass() {
    state.activeLabel = state.activeLabel === 1 ? 0 : 1;
    ui.refreshActiveClass();
  },
  onDemo() {
    state.points = makeDemoPoints(W, H);
    perceptron.reset();
  },
  onReset() {
    state.points = [];
    perceptron.reset();
  },
  onToggleTraining() {
    state.training = !state.training;
    ui.refreshTraining();
  },
  onStep() {
    // Mode pas-à-pas : une seule itération (une passe batch) par clic.
    state.training = false;
    ui.refreshTraining();
    perceptron.trainEpoch(samples(), state.learningRate);
  },
});

// --- Interaction souris : ajouter ou supprimer un point ---------------------

canvas.addEventListener("click", (event) => {
  const rect = canvas.getBoundingClientRect();
  // Le canvas peut être affiché à une taille différente de sa résolution.
  const px = ((event.clientX - rect.left) / rect.width) * W;
  const py = ((event.clientY - rect.top) / rect.height) * H;

  const hitIndex = findPointAt(px, py);
  if (hitIndex >= 0) {
    state.points.splice(hitIndex, 1); // clic sur un point existant -> suppression
  } else {
    state.points.push({ px, py, label: state.activeLabel }); // sinon -> ajout
  }
});

/** Renvoie l'index d'un point sous le curseur, ou -1. */
function findPointAt(px: number, py: number): number {
  const r = pointRadius();
  for (let i = state.points.length - 1; i >= 0; i--) {
    const p = state.points[i]!;
    if (Math.hypot(p.px - px, p.py - py) <= r) return i;
  }
  return -1;
}

// --- Boucle d'animation -----------------------------------------------------

function step(): void {
  const data = samples();

  if (state.training && data.length > 0) {
    // Chaque frame applique plusieurs itérations batch (réglé par le slider) :
    // à chaque itération on parcourt tous les points puis on met à jour la
    // droite une seule fois. On voit ainsi la frontière converger.
    for (let i = 0; i < state.speed; i++) {
      perceptron.trainEpoch(data, state.learningRate);
    }
  }

  const errors = perceptron.countErrors(data);
  ui.updateReadouts(perceptron, errors);
  draw(ctx!, W, H, state.points, perceptron);

  requestAnimationFrame(step);
}

// Démarre avec le jeu de démonstration en place.
state.points = makeDemoPoints(W, H);
requestAnimationFrame(step);
