import * as tf from "@tensorflow/tfjs";
import { ACTION_NAMES, PADDLE_SPEED, type Action } from "./breakout";
import { drawCurve, drawQBars, meanOfLast } from "./chart";
import { DqnAgent, argmax, type DqnParams } from "./dqn";
import { GameRenderer } from "./renderer";
import { ACTION_REPEAT, BUFFER_SIZE, Trainer, WARMUP, randomBaseline, type EpisodeResult, type Mode } from "./training";

const DEFAULT_PARAMS: DqnParams = {
  learningRate: 1e-3,
  gamma: 0.99,
  epsStart: 1,
  epsEnd: 0.05,
  epsDecaySteps: 50_000,
  targetPeriod: 1_000,
};
const SPEEDS = [1, 2, 4, 8, 16, 32];
const TURBO_BUDGET_MS = 25;
const HUMAN_HZ = 60;
const STORAGE_URL = "localstorage://tp9-dqn";
const PRETRAINED_URL = `${import.meta.env.BASE_URL}model/model.json`;
const MODE_LABELS: Record<Mode, string> = { train: "entraînement", demo: "démo", human: "humain" };
const MODE_HINTS: Record<Mode, string> = {
  train: `L'agent joue en ε-greedy et apprend à chaque décision (une décision tous les ${ACTION_REPEAT} pas de physique).`,
  demo: "L'agent joue en greedy (ε = 0), sans apprendre ni remplir le replay buffer.",
  human:
    "Flèches ← → ou souris sur le jeu. Votre partie ne nourrit pas le replay buffer ; les barres montrent ce que l'agent ferait.",
};

// Partagé par référence avec l'agent : les curseurs agissent en direct, sans reset.
const params: DqnParams = { ...DEFAULT_PARAMS };

const el = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const fmt = (v: number, digits = 2) =>
  v.toLocaleString("fr-FR", { minimumFractionDigits: digits, maximumFractionDigits: digits });

const gameCanvas = el<HTMLCanvasElement>("game");
const qCanvas = el<HTMLCanvasElement>("q-bars");
const scoreChart = el<HTMLCanvasElement>("chart-score");
const lossChart = el<HTMLCanvasElement>("chart-loss");
const playButton = el<HTMLButtonElement>("play");
const turboButton = el<HTMLButtonElement>("turbo");
const modeButtons: Record<Mode, HTMLButtonElement> = {
  train: el("mode-train"),
  demo: el("mode-demo"),
  human: el("mode-human"),
};
const modelButtons = ["save-local", "load-local", "export", "import", "pretrained"].map((id) => el<HTMLButtonElement>(id));
const importInput = el<HTMLInputElement>("import-files");
const modelStatus = el("model-status");
const counters = {
  episode: el("c-episode"),
  steps: el("c-steps"),
  epsilon: el("c-epsilon"),
  buffer: el("c-buffer"),
  best: el("c-best"),
  mean: el("c-mean"),
  baseline: el("c-baseline"),
  last: el("c-last"),
  backend: el("c-backend"),
  tensors: el("c-tensors"),
};

const renderer = new GameRenderer(gameCanvas);
let agent: DqnAgent;
let trainer: Trainer;
let baseline = Number.NaN;
let running = false;
let turbo = false;
let speed = SPEEDS[0];
let humanPending = 0;
let lastTime = 0;
let gameOver: EpisodeResult | null = null;
let chartedEpisodes = -1;
let chartedLosses = -1;
const scores: number[] = [];
let bestScore = Number.NaN;

// Commande humaine : la dernière source utilisée (clavier ou souris) l'emporte.
const keys = { left: false, right: false };
let control: "keyboard" | "mouse" = "keyboard";
let mouseX = 0;

interface SliderSpec {
  label: string;
  tip: string;
  min: number;
  max: number;
  step: number;
  value: number;
  format: (v: number) => string;
  onInput: (v: number) => void;
}

/** Crée un curseur étiqueté et renvoie une fonction pour le repositionner (bouton « Valeurs par défaut »). */
function addSlider(container: HTMLElement, spec: SliderSpec): (v: number) => void {
  const label = document.createElement("label");
  label.className = "slider";
  label.title = spec.tip;
  const name = document.createElement("span");
  name.textContent = spec.label;
  const output = document.createElement("output");
  const input = document.createElement("input");
  input.type = "range";
  input.min = String(spec.min);
  input.max = String(spec.max);
  input.step = String(spec.step);
  input.value = String(spec.value);
  output.textContent = spec.format(spec.value);
  input.addEventListener("input", () => {
    const v = Number(input.value);
    output.textContent = spec.format(v);
    spec.onInput(v);
  });
  label.append(name, output, input);
  container.append(label);
  return (v) => {
    input.value = String(v);
    output.textContent = spec.format(v);
    spec.onInput(v);
  };
}

interface ParamSpec {
  key: keyof DqnParams;
  label: string;
  tip: string;
  format: (v: number) => string;
  /** Soit une plage continue, soit une liste de valeurs (curseur sur l'indice). */
  range?: { min: number; max: number; step: number };
  choices?: number[];
}

const paramResets: (() => void)[] = [];
function addParam(container: HTMLElement, spec: ParamSpec): void {
  const { key, choices, format } = spec;
  const toValue = (v: number) => (choices ? choices[v] : v);
  const toSlider = (v: number) => (choices ? choices.indexOf(v) : v);
  const range = choices ? { min: 0, max: choices.length - 1, step: 1 } : spec.range!;
  const set = addSlider(container, {
    label: spec.label,
    tip: spec.tip,
    ...range,
    value: toSlider(DEFAULT_PARAMS[key]),
    format: (v) => format(toValue(v)),
    onInput: (v) => (params[key] = toValue(v)),
  });
  paramResets.push(() => set(toSlider(DEFAULT_PARAMS[key])));
}

const integer = (v: number) => v.toLocaleString("fr-FR");
const paramsSlot = el("params");
addParam(paramsSlot, {
  key: "learningRate",
  label: "Taux d'apprentissage (Adam)",
  tip: "Taille des pas de gradient : trop petit = apprentissage lent, trop grand = Q-valeurs instables.",
  choices: [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
  format: (v) => v.toExponential(0),
});
addParam(paramsSlot, {
  key: "gamma",
  label: "γ — facteur d'actualisation",
  tip: "Poids des récompenses futures : proche de 0 = agent myope, proche de 1 = agent prévoyant.",
  range: { min: 0, max: 0.999, step: 0.001 },
  format: (v) => fmt(v, 3),
});
addParam(paramsSlot, {
  key: "epsStart",
  label: "ε initial",
  tip: "Probabilité d'une action au hasard au tout début de l'entraînement.",
  range: { min: 0, max: 1, step: 0.05 },
  format: (v) => fmt(v, 2),
});
addParam(paramsSlot, {
  key: "epsEnd",
  label: "ε final",
  tip: "Exploration résiduelle une fois la décroissance terminée (ε initial = ε final : ε constant).",
  range: { min: 0, max: 1, step: 0.01 },
  format: (v) => fmt(v, 2),
});
addParam(paramsSlot, {
  key: "epsDecaySteps",
  label: "Durée de décroissance de ε (pas)",
  tip: "ε passe linéairement de sa valeur initiale à sa valeur finale sur ce nombre de décisions.",
  choices: [1_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000],
  format: integer,
});
addParam(paramsSlot, {
  key: "targetPeriod",
  label: "Recopie du réseau cible C (pas)",
  tip: "Le réseau cible reçoit une copie du réseau Q tous les C pas ; C = 1 revient à ne pas avoir de réseau cible.",
  choices: [1, 10, 100, 250, 500, 1_000, 2_000, 5_000, 10_000],
  format: integer,
});

addSlider(el("speed-slot"), {
  label: "Vitesse",
  tip: "Nombre de pas de physique simulés par image affichée (hors turbo et hors mode humain).",
  min: 0,
  max: SPEEDS.length - 1,
  step: 1,
  value: 0,
  format: (i) => `${SPEEDS[i]} pas / image`,
  onInput: (i) => (speed = SPEEDS[i]),
});

function setRunning(value: boolean): void {
  running = value;
  humanPending = 0;
  if (running) gameOver = null;
  playButton.textContent = running ? "⏸ Pause" : "▶ Lecture";
  turboButton.setAttribute("aria-pressed", String(turbo));
  turboButton.disabled = trainer.mode !== "train";
}

function setMode(mode: Mode): void {
  trainer.setMode(mode);
  for (const [m, button] of Object.entries(modeButtons)) button.setAttribute("aria-pressed", String(m === mode));
  el("mode-hint").textContent = MODE_HINTS[mode];
  if (mode !== "train") turbo = false;
  gameOver = null;
  setRunning(mode === "demo");
}

function onEpisodeEnd(result: EpisodeResult): void {
  if (result.mode === "train") {
    scores.push(result.score);
    if (!(result.score <= bestScore)) bestScore = result.score;
  }
  if (result.mode === "human") {
    gameOver = result;
    setRunning(false);
  }
}

function updateHumanAction(): void {
  let action: Action = 1;
  if (control === "keyboard") {
    if (keys.left !== keys.right) action = keys.left ? 0 : 2;
  } else {
    const diff = mouseX - trainer.env.paddleX;
    if (Math.abs(diff) > PADDLE_SPEED / 2) action = diff < 0 ? 0 : 2;
  }
  trainer.humanAction = action;
}

function simulate(dt: number): void {
  if (trainer.mode === "human") {
    // Le joueur humain joue en temps réel (60 pas / s), quel que soit l'écran.
    humanPending += (dt * HUMAN_HZ) / 1000;
    while (running && humanPending >= 1) {
      humanPending -= 1;
      updateHumanAction();
      const end = trainer.tick();
      if (end) onEpisodeEnd(end);
    }
  } else if (turbo) {
    trainer.wantQ = false;
    const stop = performance.now() + TURBO_BUDGET_MS;
    do {
      for (let i = 0; i < 20; i++) {
        const end = trainer.tick();
        if (end) onEpisodeEnd(end);
      }
    } while (performance.now() < stop);
    trainer.wantQ = true;
  } else {
    for (let i = 0; i < speed; i++) {
      const end = trainer.tick();
      if (end) onEpisodeEnd(end);
    }
  }
}

function drawQ(): void {
  const mode = trainer.mode;
  // En humain et en turbo, on interroge le réseau sur l'état affiché ; sinon on montre la dernière décision.
  if (mode === "human" || (turbo && running) || !trainer.hasQ) {
    trainer.peekQ();
    const best = argmax(trainer.q);
    const who = mode === "human" ? "l'agent jouerait" : "action greedy";
    drawQBars(qCanvas, trainer.q, best, `${who} : ${ACTION_NAMES[best]}`);
    return;
  }
  const a = trainer.action;
  const note = trainer.randomAction
    ? `action choisie : ${ACTION_NAMES[a]} (au hasard, ε)`
    : `action choisie : ${ACTION_NAMES[a]} (greedy, argmax Q)`;
  drawQBars(qCanvas, trainer.q, a, note);
}

function drawCharts(): void {
  const history = trainer.history;
  drawCurve(scoreChart, {
    label: "briques cassées par épisode",
    values: scores,
    truncated: history.map((e) => e.truncated),
    color: "#9ece6a",
    log: false,
    xValue: (i) => i + 1,
    empty: "Aucun épisode d'entraînement terminé pour l'instant",
    reference: Number.isFinite(baseline)
      ? { value: baseline, label: `politique aléatoire (${fmt(baseline, 1)})` }
      : undefined,
  });
  const losses = trainer.losses;
  drawCurve(lossChart, {
    label: "perte de Huber",
    values: losses.values,
    color: "#7aa2f7",
    log: true,
    xValue: (i) => (i + 1) * losses.updatesPerPoint,
    empty: `Pas encore d'apprentissage (warm-up : ${integer(WARMUP)} transitions dans le buffer)`,
  });
}

function message(): string | undefined {
  if (gameOver) return `Partie terminée : ${gameOver.score} briques\n▶ Lecture pour rejouer`;
  if (running) return undefined;
  if (trainer.mode === "human") return "▶ Lecture pour jouer\n← → ou souris";
  return "En pause : ▶ Lecture";
}

function render(): void {
  renderer.draw(trainer.env, message());
  drawQ();

  const mode = trainer.mode;
  counters.episode.textContent = integer(trainer.history.length + 1);
  counters.steps.textContent = integer(agent.steps);
  counters.epsilon.textContent = mode === "train" ? fmt(agent.epsilon, 3) : mode === "demo" ? "0 (greedy)" : "—";
  counters.buffer.textContent = `${integer(trainer.buffer.size)} / ${integer(BUFFER_SIZE)}`;
  counters.best.textContent = Number.isNaN(bestScore) ? "—" : String(bestScore);
  counters.mean.textContent = scores.length ? fmt(meanOfLast(scores), 1) : "—";
  const last = trainer.lastEpisode;
  counters.last.textContent = last ? `${last.score} (${MODE_LABELS[last.mode]}${last.truncated ? ", tronqué" : ""})` : "—";
  counters.tensors.textContent = integer(tf.memory().numTensors);

  if (scores.length !== chartedEpisodes || trainer.losses.values.length !== chartedLosses) {
    drawCharts();
    chartedEpisodes = scores.length;
    chartedLosses = trainer.losses.values.length;
  }
}

function frame(now: number): void {
  const dt = Math.min(100, now - lastTime);
  lastTime = now;
  if (running) simulate(dt);
  render();
  requestAnimationFrame(frame);
}

async function modelAction(work: () => Promise<string>): Promise<void> {
  for (const b of modelButtons) b.disabled = true;
  modelStatus.textContent = "…";
  try {
    modelStatus.textContent = await work();
  } catch (error) {
    modelStatus.textContent = `Échec : ${error instanceof Error ? error.message : String(error)}`;
  } finally {
    for (const b of modelButtons) b.disabled = false;
  }
}

async function loadModel(source: string | tf.io.IOHandler, what: string): Promise<string> {
  await agent.load(source);
  trainer.clearBuffer();
  setMode("demo");
  return `${what} chargé : démo en cours (ε final si l'entraînement reprend, buffer vidé).`;
}

function resetScores(): void {
  scores.length = 0;
  bestScore = Number.NaN;
  chartedEpisodes = -1;
}

function bindEvents(): void {
  playButton.addEventListener("click", () => setRunning(!running));
  turboButton.addEventListener("click", () => {
    turbo = !turbo;
    setRunning(turbo || running);
  });
  el("reset").addEventListener("click", () => {
    trainer.reset();
    resetScores();
    turbo = false;
    setMode("train");
  });
  for (const [mode, button] of Object.entries(modeButtons)) {
    button.addEventListener("click", () => setMode(mode as Mode));
  }
  el("defaults").addEventListener("click", () => {
    for (const reset of paramResets) reset();
  });

  el("save-local").addEventListener("click", () =>
    modelAction(async () => {
      await agent.save(STORAGE_URL);
      return "Réseau sauvegardé dans le navigateur (localStorage).";
    }),
  );
  el("load-local").addEventListener("click", () =>
    modelAction(() => loadModel(STORAGE_URL, "Réseau du navigateur")),
  );
  el("export").addEventListener("click", () =>
    modelAction(async () => {
      await agent.save("downloads://model");
      return "Téléchargement de model.json et model.weights.bin.";
    }),
  );
  el("import").addEventListener("click", () => importInput.click());
  importInput.addEventListener("change", () => {
    const files = [...(importInput.files ?? [])];
    importInput.value = "";
    const json = files.find((f) => f.name.endsWith(".json"));
    const weights = files.filter((f) => f.name.endsWith(".bin"));
    modelAction(async () => {
      if (!json || weights.length === 0) throw new Error("sélectionnez ensemble model.json et son fichier .bin.");
      return loadModel(tf.io.browserFiles([json, ...weights]), json.name);
    });
  });
  el("pretrained").addEventListener("click", () => modelAction(() => loadModel(PRETRAINED_URL, "Modèle pré-entraîné")));

  const onKey = (event: KeyboardEvent, down: boolean) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    if (event.key === "ArrowLeft") keys.left = down;
    else keys.right = down;
    control = "keyboard";
    if (trainer.mode === "human") event.preventDefault();
  };
  window.addEventListener("keydown", (e) => onKey(e, true));
  window.addEventListener("keyup", (e) => onKey(e, false));
  gameCanvas.addEventListener("mousemove", (event) => {
    mouseX = renderer.gameX(event);
    control = "mouse";
  });
}

async function initBackend(): Promise<void> {
  try {
    if (!(await tf.setBackend("webgl"))) await tf.setBackend("cpu");
  } catch {
    await tf.setBackend("cpu");
  }
  await tf.ready();
  counters.backend.textContent = tf.getBackend();
}

async function main(): Promise<void> {
  await initBackend();
  agent = new DqnAgent(params);
  trainer = new Trainer(agent);
  baseline = randomBaseline(30);
  counters.baseline.textContent = fmt(baseline, 1);
  bindEvents();
  setMode("train");
  lastTime = performance.now();
  requestAnimationFrame(frame);
}

main().catch((error) => {
  console.error(error);
  modelStatus.textContent = `Erreur d'initialisation : ${error instanceof Error ? error.message : String(error)}`;
});
