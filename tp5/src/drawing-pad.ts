const LINE_WIDTH = 20;

export interface DrawingPad {
  clear(): void;
}

/** Dessin à la souris ou au doigt : trait noir sur fond blanc. */
export function setupDrawingPad(canvas: HTMLCanvasElement, onChange: () => void): DrawingPad {
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  let drawing = false;
  let lastX = 0;
  let lastY = 0;

  function position(event: PointerEvent): [number, number] {
    const rect = canvas.getBoundingClientRect();
    return [
      ((event.clientX - rect.left) * canvas.width) / rect.width,
      ((event.clientY - rect.top) * canvas.height) / rect.height,
    ];
  }

  function line(x: number, y: number): void {
    ctx.strokeStyle = "#000";
    ctx.lineWidth = LINE_WIDTH;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    lastX = x;
    lastY = y;
    onChange();
  }

  function clear(): void {
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    onChange();
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) {
      return;
    }
    drawing = true;
    canvas.setPointerCapture(event.pointerId);
    [lastX, lastY] = position(event);
    line(lastX, lastY);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (drawing) {
      line(...position(event));
    }
  });
  for (const type of ["pointerup", "pointercancel"] as const) {
    canvas.addEventListener(type, () => {
      drawing = false;
    });
  }

  clear();
  return { clear };
}
