const weekday = new Intl.DateTimeFormat("fr-FR", { weekday: "long", timeZone: "UTC" });
const shortDate = new Intl.DateTimeFormat("fr-FR", { day: "2-digit", month: "2-digit", timeZone: "UTC" });
const longDate = new Intl.DateTimeFormat("fr-FR", {
  weekday: "long",
  day: "numeric",
  month: "long",
  year: "numeric",
  timeZone: "UTC",
});

export interface Form {
  showResult(tickets: number): void;
  enable(): void;
}

export function setupForm(
  dates: string[],
  values: number[],
  onSubmit: (tickets: number[]) => void,
): Form {
  const form = document.querySelector<HTMLFormElement>("#form")!;
  const fields = document.querySelector<HTMLDivElement>("#fields")!;
  const button = form.querySelector<HTMLButtonElement>("button[type=submit]")!;
  const result = document.querySelector<HTMLOutputElement>("#result")!;

  const inputs = dates.map((iso, i) => {
    const day = new Date(iso);
    const label = document.createElement("label");
    label.className = "field";
    label.htmlFor = `day-${i}`;
    label.innerHTML = `<span class="weekday"></span><span class="date"></span>`;
    label.querySelector(".weekday")!.textContent = weekday.format(day);
    label.querySelector(".date")!.textContent = shortDate.format(day);

    const input = document.createElement("input");
    input.id = `day-${i}`;
    input.type = "number";
    input.min = "0";
    input.step = "1";
    input.required = true;
    input.value = String(values[i]);
    fields.append(label, input);
    return input;
  });

  const last = new Date(dates[dates.length - 1]);
  const tomorrow = new Date(last.getTime() + 24 * 3600 * 1000);

  // La validation native (required, min, step) bloque la soumission des valeurs invalides.
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    onSubmit(inputs.map((input) => input.valueAsNumber));
  });

  return {
    showResult(tickets) {
      result.innerHTML = `<span>Prévision pour le <span class="next-day"></span> :</span><strong></strong>`;
      result.querySelector(".next-day")!.textContent = longDate.format(tomorrow);
      result.querySelector("strong")!.textContent = `${Math.max(0, Math.round(tickets))} tickets`;
    },
    enable() {
      button.disabled = false;
    },
  };
}
