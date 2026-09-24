import { Component, effect, inject, signal } from '@angular/core';
import { DigitClassifier } from './digit-classifier';
import { DigitPreview } from './digit-preview';
import { DrawingPad } from './drawing-pad';
import { PredictionPanel } from './prediction-panel';
import { preprocess } from './preprocess';

@Component({
  imports: [DrawingPad, DigitPreview, PredictionPanel],
  selector: 'app-root',
  templateUrl: './app.html',
})
export class App {
  private readonly classifier = inject(DigitClassifier);

  protected readonly status = this.classifier.status;
  protected readonly pixels = signal<Float32Array<ArrayBuffer> | null>(null);
  protected readonly probabilities = signal<number[] | null>(null);

  private canvas: HTMLCanvasElement | null = null;
  private dirty = false;
  private scheduled = false;

  constructor() {
    // Relance la reconnaissance si l'utilisateur a dessiné avant la fin du chargement du modèle.
    effect(() => {
      if (this.status() === 'ready' && this.canvas) {
        this.schedule();
      }
    });
  }

  protected onDrawn(canvas: HTMLCanvasElement): void {
    this.canvas = canvas;
    this.schedule();
  }

  protected clear(pad: DrawingPad): void {
    pad.clear();
    this.canvas = null;
    this.dirty = false;
    this.pixels.set(null);
    this.probabilities.set(null);
  }

  // Au plus une inférence par frame et une seule en cours à la fois.
  private schedule(): void {
    this.dirty = true;
    if (!this.scheduled) {
      this.scheduled = true;
      requestAnimationFrame(() => void this.recognize());
    }
  }

  private async recognize(): Promise<void> {
    while (this.dirty && this.canvas) {
      this.dirty = false;
      const pixels = preprocess(this.canvas);
      this.pixels.set(pixels);
      if (pixels && this.status() === 'ready') {
        const probabilities = await this.classifier.predict(pixels);
        if (this.canvas) {
          this.probabilities.set(probabilities);
        }
      }
    }
    this.scheduled = false;
  }
}
