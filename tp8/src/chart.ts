// Courbes par épisode : points bruts, moyenne glissante, épisodes tronqués, ligne de référence.

const MARGIN = { top: 30, right: 16, bottom: 26, left: 56 };
const TEXT = "#9a9ab0";
const GRID = "#3a3a52";
const TRUNCATED = "#f7768e";
const REFERENCE = "#e6e6f0";
const WINDOW = 20;
const number = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0 });

export interface EpisodeSeries {
  label: string;
  values: number[];
  truncated: boolean[];
  color: string;
  log: boolean;
  reference?: { value: number; label: string };
}

function niceStep(raw: number): number {
  const pow = 10 ** Math.floor(Math.log10(raw));
  const f = raw / pow;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * pow;
}

function movingAverage(values: number[], window: number): number[] {
  const out: number[] = [];
  let sum = 0;
  values.forEach((v, i) => {
    sum += v;
    if (i >= window) sum -= values[i - window];
    out.push(sum / Math.min(i + 1, window));
  });
  return out;
}

export function drawEpisodeChart(canvas: HTMLCanvasElement, series: EpisodeSeries): void {
  const ctx = canvas.getContext("2d")!;
  const { values, truncated, color, log, reference } = series;
  const width = canvas.width - MARGIN.left - MARGIN.right;
  const height = canvas.height - MARGIN.top - MARGIN.bottom;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.font = "12px system-ui, sans-serif";
  if (values.length === 0) {
    ctx.fillStyle = TEXT;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("Aucun épisode terminé pour l'instant", canvas.width / 2, canvas.height / 2);
    return;
  }

  // Boucle plutôt que Math.min(...values) : le spread déborde sur de très longs historiques.
  let vMin = reference?.value ?? Infinity;
  let vMax = reference?.value ?? -Infinity;
  for (const v of values) {
    if (v < vMin) vMin = v;
    if (v > vMax) vMax = v;
  }
  let lo: number;
  let hi: number;
  let ticks: number[];
  let f: (v: number) => number;
  if (log) {
    f = (v) => Math.log10(Math.max(1, v));
    lo = Math.floor(f(vMin));
    hi = Math.max(lo + 1, Math.ceil(f(vMax)));
    ticks = [];
    for (let k = lo; k <= hi; k++) ticks.push(10 ** k);
  } else {
    f = (v) => v;
    const step = niceStep(Math.max(1, vMax - vMin) / 5);
    lo = Math.floor(vMin / step) * step;
    hi = Math.max(lo + step, Math.ceil(vMax / step) * step);
    ticks = [];
    for (let v = lo; v <= hi + step / 2; v += step) ticks.push(v);
  }
  const n = values.length;
  const x = (i: number) => MARGIN.left + (n === 1 ? 0.5 : i / (n - 1)) * width;
  const y = (v: number) => MARGIN.top + height - ((f(v) - lo) / (hi - lo)) * height;

  ctx.lineWidth = 1;
  ctx.strokeStyle = GRID;
  ctx.fillStyle = TEXT;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const v of ticks) {
    ctx.beginPath();
    ctx.moveTo(MARGIN.left, y(v));
    ctx.lineTo(MARGIN.left + width, y(v));
    ctx.stroke();
    ctx.fillText(number.format(v), MARGIN.left - 6, y(v));
  }

  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const xStep = niceStep(Math.max(1, n / 8));
  for (let e = xStep; e <= n; e += xStep) ctx.fillText(number.format(e), x(e - 1), MARGIN.top + height + 6);

  ctx.globalAlpha = 0.35;
  ctx.fillStyle = color;
  values.forEach((v, i) => {
    if (!truncated[i]) ctx.fillRect(x(i) - 1, y(v) - 1, 2, 2);
  });
  ctx.globalAlpha = 0.8;
  ctx.fillStyle = TRUNCATED;
  values.forEach((v, i) => {
    if (truncated[i]) ctx.fillRect(x(i) - 1.5, y(v) - 1.5, 3, 3);
  });
  ctx.globalAlpha = 1;

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  movingAverage(values, WINDOW).forEach((v, i) => (i === 0 ? ctx.moveTo(x(i), y(v)) : ctx.lineTo(x(i), y(v))));
  ctx.stroke();

  if (reference) {
    ctx.strokeStyle = REFERENCE;
    ctx.lineWidth = 1.2;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(MARGIN.left, y(reference.value));
    ctx.lineTo(MARGIN.left + width, y(reference.value));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  const legend: { label: string; color: string; kind: "line" | "dot" | "dash" }[] = [
    { label: `${series.label} (brut)`, color, kind: "dot" },
    { label: `moyenne sur ${WINDOW} épisodes`, color, kind: "line" },
    { label: "épisode tronqué", color: TRUNCATED, kind: "dot" },
  ];
  if (reference) legend.push({ label: reference.label, color: REFERENCE, kind: "dash" });

  let lx = MARGIN.left;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  for (const item of legend) {
    ctx.strokeStyle = item.color;
    ctx.fillStyle = item.color;
    if (item.kind === "dot") {
      ctx.fillRect(lx + 8, 10, 4, 4);
    } else {
      ctx.lineWidth = item.kind === "line" ? 2 : 1.2;
      ctx.setLineDash(item.kind === "dash" ? [5, 4] : []);
      ctx.beginPath();
      ctx.moveTo(lx, 12);
      ctx.lineTo(lx + 20, 12);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.fillStyle = TEXT;
    ctx.fillText(item.label, lx + 26, 12);
    lx += 26 + ctx.measureText(item.label).width + 22;
  }
}
