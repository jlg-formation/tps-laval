import { QAgent, type AgentParams } from "./agent";
import { drawEpisodeChart } from "./chart";
import { ACTION_ARROWS, ACTION_NAMES, generateMaze, shortestPathLength, type Rewards } from "./maze";
import { MazeRenderer, PALETTE_GRADIENT } from "./renderer";
import { Trainer, type Transition } from "./training";

const DEFAULT_PARAMS: AgentParams = { alpha: 0.1, gamma: 0.99, epsilon0: 1, epsMin: 0.05, decay: 0.99 };
const DEFAULT_REWARDS: Rewards = { goal: 100, step: -1, wall: -5 };
const DEFAULT_SIZE = 10;
const SPEEDS = [0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200];
const DEFAULT_SPEED_INDEX = 5;
const TURBO_BUDGET_MS = 12;
// Sans plafond, un petit labyrinthe enchaîne des milliers d'épisodes par seconde et écrase la courbe.
const turboMaxEpisodes = () => Math.ceil((size * size) / 10);

// Partagés par référence avec l'agent et l'environnement : les curseurs agissent en direct.
const params: AgentParams = { ...DEFAULT_PARAMS };
const rewards: Rewards = { ...DEFAULT_REWARDS };

const el = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const fmt = (v: number, digits = 2) =>
  v.toLocaleString("fr-FR", { minimumFractionDigits: digits, maximumFractionDigits: digits });

const mazeCanvas = el<HTMLCanvasElement>("maze");
const tooltip = el("tooltip");
const transitionEl = el("transition");
const playButton = el<HTMLButtonElement>("play");
const turboButton = el<HTMLButtonElement>("turbo");
const stepsChart = el<HTMLCanvasElement>("chart-steps");
const rewardChart = el<HTMLCanvasElement>("chart-reward");
const counters = {
  episode: el("c-episode"),
  steps: el("c-steps"),
  epsilon: el("c-epsilon"),
  last: el("c-last"),
  optimal: el("c-optimal"),
  vMin: el("v-min"),
  vMax: el("v-max"),
};
el("gradient").style.background = PALETTE_GRADIENT;

const renderer = new MazeRenderer(mazeCanvas);
let trainer: Trainer;
let optimal = 0;
let running = false;
let turbo = false;
let speed = SPEEDS[DEFAULT_SPEED_INDEX];
let pendingSteps = 0;
let hover: number | null = null;
let dirty = true;
let chartedEpisodes = -1;

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

interface ParamSpec<K extends string> {
  key: K;
  label: string;
  tip: string;
  min: number;
  max: number;
  step: number;
  digits: number;
  onInput?: (v: number) => void;
}

const paramSetters: (() => void)[] = [];
function addParams<K extends string>(
  container: HTMLElement,
  target: Record<K, number>,
  defaults: Record<K, number>,
  specs: ParamSpec<K>[],
): void {
  for (const { key, digits, onInput, ...spec } of specs) {
    const set = addSlider(container, {
      ...spec,
      value: defaults[key],
      format: (v) => fmt(v, digits),
      onInput: (v) => {
        target[key] = v;
        onInput?.(v);
      },
    });
    paramSetters.push(() => set(defaults[key]));
  }
}

addParams(el("params"), params, DEFAULT_PARAMS, [
  {
    key: "alpha",
    label: "α — taux d'apprentissage",
    tip: "Part de l'écart (cible − Q) corrigée à chaque mise à jour : 0 = n'apprend rien, 1 = remplace Q par la cible.",
    min: 0.01,
    max: 1,
    step: 0.01,
    digits: 2,
  },
  {
    key: "gamma",
    label: "γ — facteur d'actualisation",
    tip: "Poids des récompenses futures : proche de 0 = agent myope, proche de 1 = agent prévoyant.",
    min: 0.5,
    max: 1,
    step: 0.005,
    digits: 3,
  },
  {
    key: "epsilon0",
    label: "ε initial",
    tip: "Probabilité d'une action au hasard au début (exploration). Le modifier remet ε courant à cette valeur.",
    min: 0,
    max: 1,
    step: 0.05,
    digits: 2,
    onInput: (v) => (trainer.agent.epsilon = v),
  },
  {
    key: "epsMin",
    label: "ε minimum",
    tip: "Plancher de ε : l'agent garde toujours un peu d'exploration.",
    min: 0,
    max: 0.5,
    step: 0.01,
    digits: 2,
    onInput: (v) => (trainer.agent.epsilon = Math.max(trainer.agent.epsilon, v)),
  },
  {
    key: "decay",
    label: "Décroissance de ε",
    tip: "À la fin de chaque épisode, ε est multiplié par ce facteur (sans descendre sous ε minimum).",
    min: 0.9,
    max: 0.999,
    step: 0.001,
    digits: 3,
  },
]);

addParams(el("rewards"), rewards, DEFAULT_REWARDS, [
  {
    key: "goal",
    label: "Sortie",
    tip: "Récompense reçue en atteignant G ; l'épisode se termine.",
    min: 0,
    max: 500,
    step: 10,
    digits: 0,
  },
  {
    key: "step",
    label: "Pas",
    tip: "Récompense de chaque déplacement : négative, elle pousse l'agent à trouver un chemin court.",
    min: -10,
    max: 0,
    step: 0.5,
    digits: 1,
  },
  {
    key: "wall",
    label: "Mur",
    tip: "Récompense (à la place de celle du pas) quand l'agent se cogne à un mur ; il reste sur place.",
    min: -20,
    max: 0,
    step: 1,
    digits: 0,
  },
]);

addSlider(el("speed-slot"), {
  label: "Vitesse",
  tip: "Nombre de pas d'apprentissage par image affichée (hors mode turbo).",
  min: 0,
  max: SPEEDS.length - 1,
  step: 1,
  value: DEFAULT_SPEED_INDEX,
  format: (i) => `${fmt(SPEEDS[i], SPEEDS[i] < 1 ? 2 : 0)} pas / image`,
  onInput: (i) => (speed = SPEEDS[i]),
});

let size = DEFAULT_SIZE;
addSlider(el("size-slot"), {
  label: "Taille (cellules par côté)",
  tip: "Change la taille et génère un nouveau labyrinthe ; l'agent repart de zéro.",
  min: 5,
  max: 25,
  step: 1,
  value: DEFAULT_SIZE,
  format: (n) => `${n} × ${n}`,
  onInput: (n) => {
    size = n;
    newMaze();
  },
});

function invalidate(): void {
  dirty = true;
  chartedEpisodes = -1;
}

function setRunning(value: boolean): void {
  running = value;
  pendingSteps = 0;
  playButton.textContent = running ? "⏸ Pause" : "▶ Lecture";
  turboButton.setAttribute("aria-pressed", String(turbo));
}

function clearTransition(): void {
  transitionEl.textContent = "Cliquez sur « Pas à pas » pour voir le détail d'une mise à jour de Q.";
}

function newMaze(): void {
  const maze = generateMaze(size);
  trainer = new Trainer(maze, new QAgent(maze.width * maze.width, params), rewards);
  optimal = shortestPathLength(maze);
  renderer.setMaze(maze);
  hover = null;
  tooltip.hidden = true;
  setRunning(false);
  clearTransition();
  invalidate();
}

function showTransition(t: Transition): void {
  const w = trainer.maze.width;
  const pos = (s: number) => `(${s % w}, ${Math.floor(s / w)})`;
  const u = t.update;
  const p = (v: number) => (v < 0 ? `(${fmt(v)})` : fmt(v));
  const lines = [
    `s  = ${pos(t.s)}      a = ${ACTION_ARROWS[t.a]} ${ACTION_NAMES[t.a]}`,
    `s' = ${t.terminal ? "sortie G (terminal)" : pos(t.next)}${t.hitWall ? "   (mur : l'agent reste sur place)" : ""}`,
    `r  = ${fmt(t.r)}`,
    "",
    t.terminal
      ? `cible = r = ${fmt(u.target)}   (s' terminal : pas de γ·max Q)`
      : `cible = r + γ·max Q(s', ·) = ${fmt(t.r)} + ${fmt(params.gamma, 3)} × ${p(u.maxNext)} = ${fmt(u.target)}`,
    "Q(s, a) ← Q(s, a) + α·(cible − Q(s, a))",
    `        = ${fmt(u.before)} + ${fmt(params.alpha)} × (${fmt(u.target)} − ${p(u.before)})`,
    `        = ${fmt(u.after)}`,
  ];
  if (t.ended) {
    lines.push(
      "",
      t.ended.truncated
        ? `Épisode tronqué après ${t.ended.steps} pas (limite atteinte) : retour en S.`
        : `Sortie atteinte en ${t.ended.steps} pas : nouvel épisode depuis S.`,
    );
  }
  transitionEl.textContent = lines.join("\n");
}

function updateTooltip(i: number): void {
  const { maze, agent } = trainer;
  const w = maze.width;
  const line = (text: string, className?: string) => {
    const div = document.createElement("div");
    div.textContent = text;
    if (className) div.className = className;
    return div;
  };
  const rows = [line(`Case (${i % w}, ${Math.floor(i / w)})`, "title")];
  if (maze.walls[i]) {
    rows.push(line("Mur"));
  } else if (i === maze.goal) {
    rows.push(line("Sortie : état terminal, pas de valeur Q"));
  } else {
    const best = agent.greedyAction(i);
    for (let a = 0; a < 4; a++) {
      rows.push(line(`${ACTION_ARROWS[a]} ${ACTION_NAMES[a]} : ${fmt(agent.q[i * 4 + a])}`, a === best ? "best" : undefined));
    }
    rows.push(line(`V(s) = max Q = ${fmt(agent.maxQ(i))}`, "value"));
    if (!agent.visited[i]) rows.push(line("jamais visitée", "value"));
  }
  tooltip.replaceChildren(...rows);
}

function drawCharts(): void {
  const h = trainer.history;
  const truncated = h.map((e) => e.truncated);
  drawEpisodeChart(stepsChart, {
    label: "pas par épisode",
    values: h.map((e) => e.steps),
    truncated,
    color: "#7aa2f7",
    log: true,
    reference: { value: optimal, label: `plus court chemin (${optimal} pas)` },
  });
  const bestReward = rewards.goal + rewards.step * (optimal - 1);
  drawEpisodeChart(rewardChart, {
    label: "récompense totale",
    values: h.map((e) => e.totalReward),
    truncated,
    color: "#9ece6a",
    log: false,
    reference: { value: bestReward, label: `récompense optimale (${fmt(bestReward, 0)})` },
  });
}

function render(): void {
  const range = renderer.draw(trainer.agent, trainer.position, hover);
  counters.vMin.textContent = range ? fmt(range.min, 1) : "—";
  counters.vMax.textContent = range ? fmt(range.max, 1) : "—";
  const history = trainer.history;
  const last = history[history.length - 1];
  counters.episode.textContent = String(history.length + 1);
  counters.steps.textContent = `${trainer.steps} / ${trainer.maxSteps}`;
  counters.epsilon.textContent = fmt(trainer.agent.epsilon, 3);
  counters.last.textContent = last ? `${last.steps} pas${last.truncated ? " (tronqué)" : ""}` : "—";
  counters.optimal.textContent = `${optimal} pas`;
  if (hover !== null) updateTooltip(hover);
  if (history.length !== chartedEpisodes) {
    drawCharts();
    chartedEpisodes = history.length;
  }
}

function frame(): void {
  if (running) {
    if (turbo) {
      const end = performance.now() + TURBO_BUDGET_MS;
      const stopAt = trainer.history.length + turboMaxEpisodes();
      do {
        for (let i = 0; i < 500 && trainer.history.length < stopAt; i++) trainer.tick();
      } while (trainer.history.length < stopAt && performance.now() < end);
    } else {
      pendingSteps += speed;
      while (pendingSteps >= 1) {
        trainer.tick();
        pendingSteps -= 1;
      }
    }
    dirty = true;
  }
  if (dirty) {
    render();
    dirty = false;
  }
  requestAnimationFrame(frame);
}

playButton.addEventListener("click", () => {
  if (!running) clearTransition();
  setRunning(!running);
});
turboButton.addEventListener("click", () => {
  turbo = !turbo;
  setRunning(turbo || running);
});
el("step").addEventListener("click", () => {
  setRunning(false);
  showTransition(trainer.tick());
  dirty = true;
});
el("reset").addEventListener("click", () => {
  trainer.reset();
  setRunning(false);
  clearTransition();
  invalidate();
});
el("new-maze").addEventListener("click", newMaze);
el("defaults").addEventListener("click", () => {
  for (const reset of paramSetters) reset();
  invalidate();
});

mazeCanvas.addEventListener("mousemove", (event) => {
  hover = renderer.cellAt(event);
  tooltip.hidden = hover === null;
  tooltip.style.left = `${event.offsetX + 16}px`;
  tooltip.style.top = `${event.offsetY + 16}px`;
  // Moitié droite : la bulle passe à gauche du curseur pour ne pas sortir du cadre.
  tooltip.classList.toggle("flip", event.offsetX > mazeCanvas.clientWidth / 2);
  dirty = true;
});
mazeCanvas.addEventListener("mouseleave", () => {
  hover = null;
  tooltip.hidden = true;
  dirty = true;
});

newMaze();
requestAnimationFrame(frame);
