import { COLORS, type Bounds, type Latent, type LatentPoint } from "./latent";

const POINT_RADIUS = 2;
const CURSOR_RADIUS = 7;
const KEY_STEP = 0.02;
const KEY_STEP_FAST = 0.1;

export interface Plane {
  readonly position: Latent;
  /** Image affichée en fond à la place du nuage de points ; null pour revenir au nuage. */
  setBackground(image: CanvasImageSource | null): void;
}

export function setupPlane(
  canvas: HTMLCanvasElement,
  bounds: Bounds,
  points: readonly LatentPoint[],
  onMove: (z: Latent) => void,
): Plane {
  const ctx = canvas.getContext("2d")!;
  const { width, height } = canvas;
  const [x0, x1] = bounds.z1;
  const [y0, y1] = bounds.z2;
  const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

  // z₂ croît vers le haut, comme sur un repère mathématique.
  const toCanvas = ([z1, z2]: Latent): [number, number] => [
    ((z1 - x0) / (x1 - x0)) * width,
    (1 - (z2 - y0) / (y1 - y0)) * height,
  ];
  const toLatent = (x: number, y: number): Latent => [
    clamp(x0 + (x / width) * (x1 - x0), x0, x1),
    clamp(y0 + (1 - y / height) * (y1 - y0), y0, y1),
  ];

  let position: Latent = [clamp(0, x0, x1), clamp(0, y0, y1)];
  let background: CanvasImageSource | null = null;

  function drawAxes(): void {
    const [ox, oy] = toCanvas([0, 0]);
    ctx.save();
    ctx.strokeStyle = "rgba(230, 230, 240, 0.35)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(ox, 0);
    ctx.lineTo(ox, height);
    ctx.moveTo(0, oy);
    ctx.lineTo(width, oy);
    ctx.stroke();
    ctx.fillStyle = "rgba(230, 230, 240, 0.7)";
    ctx.font = "13px system-ui, sans-serif";
    ctx.fillText("z₁", width - 20, oy - 6);
    ctx.fillText("z₂", ox + 6, 16);
    ctx.restore();
  }

  function draw(): void {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, width, height);
    if (background) {
      ctx.drawImage(background, 0, 0, width, height);
    } else {
      ctx.globalAlpha = 0.7;
      for (const [z1, z2, label] of points) {
        const [x, y] = toCanvas([z1, z2]);
        ctx.fillStyle = COLORS[label];
        ctx.beginPath();
        ctx.arc(x, y, POINT_RADIUS, 0, 2 * Math.PI);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
    drawAxes();

    const [x, y] = toCanvas(position);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#10101a";
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(x, y, CURSOR_RADIUS, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  }

  function moveTo(z: Latent): void {
    position = z;
    draw();
    onMove(position);
  }

  function fromPointer(event: PointerEvent): Latent {
    const rect = canvas.getBoundingClientRect();
    return toLatent(
      ((event.clientX - rect.left) / rect.width) * width,
      ((event.clientY - rect.top) / rect.height) * height,
    );
  }

  canvas.addEventListener("pointerdown", (event) => {
    canvas.setPointerCapture(event.pointerId);
    canvas.focus();
    moveTo(fromPointer(event));
  });
  canvas.addEventListener("pointermove", (event) => {
    if (canvas.hasPointerCapture(event.pointerId)) {
      moveTo(fromPointer(event));
    }
  });

  const directions: Record<string, Latent> = {
    ArrowLeft: [-1, 0],
    ArrowRight: [1, 0],
    ArrowUp: [0, 1],
    ArrowDown: [0, -1],
  };
  canvas.addEventListener("keydown", (event) => {
    const direction = directions[event.key];
    if (!direction) {
      return;
    }
    event.preventDefault();
    const step = event.shiftKey ? KEY_STEP_FAST : KEY_STEP;
    moveTo([
      clamp(position[0] + direction[0] * step * (x1 - x0), x0, x1),
      clamp(position[1] + direction[1] * step * (y1 - y0), y0, y1),
    ]);
  });

  draw();

  return {
    get position() {
      return position;
    },
    setBackground(image) {
      background = image;
      draw();
    },
  };
}
