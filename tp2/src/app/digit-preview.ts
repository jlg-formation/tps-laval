import { Component, ElementRef, afterRenderEffect, input, viewChild } from '@angular/core';

const SIZE = 28;

@Component({
  selector: 'app-digit-preview',
  template: `
    <canvas
      #canvas
      [width]="size"
      [height]="size"
      class="size-28 rounded border border-slate-300 bg-black"
      style="image-rendering: pixelated"
      aria-label="Aperçu 28×28 de l'image envoyée au modèle"
    ></canvas>
  `,
})
export class DigitPreview {
  readonly pixels = input<Float32Array | null>(null);

  protected readonly size = SIZE;
  private readonly canvas = viewChild.required<ElementRef<HTMLCanvasElement>>('canvas');

  constructor() {
    afterRenderEffect(() => {
      const pixels = this.pixels();
      const ctx = this.canvas().nativeElement.getContext('2d')!;
      const image = ctx.createImageData(SIZE, SIZE);
      pixels?.forEach((v, i) => {
        const gray = Math.round(v * 255);
        image.data.set([gray, gray, gray], i * 4);
      });
      for (let i = 3; i < image.data.length; i += 4) {
        image.data[i] = 255;
      }
      ctx.putImageData(image, 0, 0);
    });
  }
}
