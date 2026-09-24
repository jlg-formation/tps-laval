import type { Metric, ModelInfo } from "./predictor";

const MARGIN = { top: 32, right: 16, bottom: 28, left: 44 };
const month = new Intl.DateTimeFormat("fr-FR", { month: "short", timeZone: "UTC" });

interface Line {
  label: string;
  values: number[];
  color: string;
  width: number;
  dash?: number[];
}

export function drawChart(canvas: HTMLCanvasElement, info: ModelInfo): void {
  const ctx = canvas.getContext("2d")!;
  const { dates, actual, predicted, lastWeek } = info.test;
  const lines: Line[] = [
    { label: "Réel", values: actual, color: "#30303a", width: 1.5 },
    { label: `${info.cell.toUpperCase()} (prédit)`, values: predicted, color: "#3b6fd8", width: 2 },
    { label: "Semaine précédente (J-7)", values: lastWeek, color: "#e08a2e", width: 1.2, dash: [4, 3] },
  ];

  const width = canvas.width - MARGIN.left - MARGIN.right;
  const height = canvas.height - MARGIN.top - MARGIN.bottom;
  const step = 20;
  const yMax = Math.ceil(Math.max(...lines.flatMap((l) => l.values)) / step) * step;
  const x = (i: number) => MARGIN.left + (i / (dates.length - 1)) * width;
  const y = (v: number) => MARGIN.top + height - (Math.max(0, v) / yMax) * height;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.font = "12px system-ui, sans-serif";
  ctx.lineWidth = 1;

  ctx.fillStyle = "#666";
  ctx.strokeStyle = "#dcdce6";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let v = 0; v <= yMax; v += step) {
    ctx.beginPath();
    ctx.moveTo(MARGIN.left, y(v));
    ctx.lineTo(MARGIN.left + width, y(v));
    ctx.stroke();
    ctx.fillText(String(v), MARGIN.left - 6, y(v));
  }

  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  dates.forEach((iso, i) => {
    if (iso.endsWith("-01")) {
      ctx.beginPath();
      ctx.moveTo(x(i), MARGIN.top);
      ctx.lineTo(x(i), MARGIN.top + height);
      ctx.stroke();
      ctx.fillText(month.format(new Date(iso)), x(i) + 3, MARGIN.top + height + 8);
    }
  });

  let legendX = MARGIN.left;
  for (const line of lines) {
    ctx.strokeStyle = line.color;
    ctx.lineWidth = line.width;
    ctx.setLineDash(line.dash ?? []);
    ctx.beginPath();
    line.values.forEach((v, i) => (i === 0 ? ctx.moveTo(x(i), y(v)) : ctx.lineTo(x(i), y(v))));
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(legendX, 12);
    ctx.lineTo(legendX + 20, 12);
    ctx.stroke();
    ctx.fillStyle = "#30303a";
    ctx.textBaseline = "middle";
    ctx.fillText(line.label, legendX + 26, 12);
    legendX += 26 + ctx.measureText(line.label).width + 24;
  }
  ctx.setLineDash([]);
}

export function renderMetrics(tbody: HTMLTableSectionElement, metrics: Metric[]): void {
  const best = Math.min(...metrics.map((m) => m.mae));
  tbody.replaceChildren(
    ...metrics.map((m) => {
      const row = document.createElement("tr");
      row.classList.toggle("best", m.mae === best);
      for (const text of [m.name, m.mae.toFixed(1), m.rmse.toFixed(1)]) {
        const cell = document.createElement("td");
        cell.textContent = text;
        row.append(cell);
      }
      return row;
    }),
  );
}
