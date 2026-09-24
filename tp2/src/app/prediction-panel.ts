import { DecimalPipe } from '@angular/common';
import { Component, computed, input } from '@angular/core';
import { ClassifierStatus } from './digit-classifier';

@Component({
  selector: 'app-prediction-panel',
  template: `
    @switch (status()) {
      @case ('loading') {
        <p class="text-slate-600">Chargement du modèle…</p>
      }
      @case ('error') {
        <p class="text-red-700" role="alert">
          Impossible de charger le modèle. Lancez d'abord <code>mise run tp2-train</code> ou <code>mise run tp3-train</code>.
        </p>
      }
      @default {
        <div aria-live="polite">
          @if (best() !== null) {
            <p class="text-slate-600">Chiffre reconnu</p>
            <p class="text-8xl font-bold leading-none text-indigo-700">{{ best() }}</p>
          } @else {
            <p class="text-slate-600">Dessinez un chiffre pour lancer la reconnaissance.</p>
          }
        </div>
      }
    }

    <ul class="mt-6 space-y-1" aria-label="Probabilités par chiffre">
      @for (p of bars(); track $index) {
        <li class="flex items-center gap-2 text-sm">
          <span class="w-4 font-mono font-semibold">{{ $index }}</span>
          <span class="h-4 flex-1 overflow-hidden rounded bg-slate-200" aria-hidden="true">
            <span
              class="block h-full"
              [class]="$index === best() ? 'bg-indigo-600' : 'bg-slate-500'"
              [style.width.%]="p * 100"
            ></span>
          </span>
          <span class="w-14 text-right font-mono tabular-nums">{{ p * 100 | number: '1.1-1' }} %</span>
        </li>
      }
    </ul>
  `,
  imports: [DecimalPipe],
})
export class PredictionPanel {
  readonly status = input.required<ClassifierStatus>();
  readonly probabilities = input<number[] | null>(null);

  protected readonly bars = computed(() => this.probabilities() ?? Array<number>(10).fill(0));
  protected readonly best = computed(() => {
    const p = this.probabilities();
    return p ? p.indexOf(Math.max(...p)) : null;
  });
}
