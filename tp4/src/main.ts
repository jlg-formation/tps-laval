import { drawChart, renderMetrics } from "./chart";
import { loadInfo, loadPredictor } from "./predictor";
import { setupForm } from "./ui";

const status = document.querySelector<HTMLParagraphElement>("#status")!;

async function main(): Promise<void> {
  const [info, predict] = await Promise.all([loadInfo(), loadPredictor()]);

  drawChart(document.querySelector<HTMLCanvasElement>("#chart")!, info);
  renderMetrics(document.querySelector<HTMLTableSectionElement>("#metrics")!, info.metrics);

  const form = setupForm(info.lastWindow.dates, info.lastWindow.values, async (tickets) => {
    form.showResult(await predict(tickets));
  });
  form.enable();
  status.textContent =
    `Modèle ${info.cell.toUpperCase()} chargé : ${info.window} jours en entrée, ` +
    `${info.hidden} neurones cachés.`;
}

main().catch((error: unknown) => {
  console.error(error);
  status.textContent = "Échec du chargement du modèle. Lancez d'abord « mise run tp4-train ».";
  status.classList.add("error");
});
