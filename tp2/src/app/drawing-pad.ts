import { Component, ElementRef, afterNextRender, output, viewChild } from '@angular/core';

const SIZE = 280;
const LINE_WIDTH = 20;

@Component({
  selector: 'app-drawing-pad',
  template: `
    <canvas
      #canvas
      [width]="size"
      [height]="size"
      class="block cursor-crosshair rounded-lg border border-slate-300 bg-white shadow-sm"
      aria-label="Zone de dessin : tracez un chiffre de 0 à 9 avec la souris"
      (mousedown)="start($event)"
      (mousemove)="move($event)"
      (mouseup)="stop()"
      (mouseleave)="stop()"
    ></canvas>
  `,
})
export class DrawingPad {
  readonly changed = output<HTMLCanvasElement>();

  protected readonly size = SIZE;
  private readonly canvas = viewChild.required<ElementRef<HTMLCanvasElement>>('canvas');
  private drawing = false;
  private lastX = 0;
  private lastY = 0;

  constructor() {
    afterNextRender(() => this.clear());
  }

  clear(): void {
    const ctx = this.context();
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, SIZE, SIZE);
  }

  protected start(event: MouseEvent): void {
    if (event.button !== 0) {
      return;
    }
    this.drawing = true;
    [this.lastX, this.lastY] = this.position(event);
    this.line(this.lastX, this.lastY);
  }

  protected move(event: MouseEvent): void {
    if (this.drawing) {
      this.line(...this.position(event));
    }
  }

  protected stop(): void {
    this.drawing = false;
  }

  private line(x: number, y: number): void {
    const ctx = this.context();
    ctx.strokeStyle = '#000';
    ctx.lineWidth = LINE_WIDTH;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(this.lastX, this.lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    this.lastX = x;
    this.lastY = y;
    this.changed.emit(this.canvas().nativeElement);
  }

  private position(event: MouseEvent): [number, number] {
    const canvas = this.canvas().nativeElement;
    const rect = canvas.getBoundingClientRect();
    return [
      ((event.clientX - rect.left) * canvas.width) / rect.width,
      ((event.clientY - rect.top) * canvas.height) / rect.height,
    ];
  }

  private context(): CanvasRenderingContext2D {
    return this.canvas().nativeElement.getContext('2d', { willReadFrequently: true })!;
  }
}
