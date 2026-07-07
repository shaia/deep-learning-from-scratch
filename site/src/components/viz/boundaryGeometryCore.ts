// Interactive decision-boundary geometry — vanilla TypeScript + Canvas.
// Companion to Module 00's "what the weights mean" section. It makes the algebra
// z = w·x + b geometric: the weight vector w is the *normal* to the boundary,
// the bias b slides that boundary off the origin, and z / ‖w‖ is the *signed
// distance* from a point to the line. Drag the w arrow and the probe point and
// watch all three facts hold at once.
//
// Follows docs/conventions/viz-style.md, same lifecycle as perceptronCore.ts:
//   • semantic colors read live from the theme (CSS variables), never hardcoded
//   • repaint on `themechange`
//   • deterministic start + a reset control; pointer-drag interaction
//   • labels use the math-notation symbols (w, b, z)

export interface GeometryState {
  w0: number; w1: number; b: number;
  px: number; py: number;
  z: number;       // pre-activation at the probe: w·p + b
  normW: number;   // ‖w‖
  dist: number;    // signed distance z / ‖w‖
}

interface Palette {
  surface: string; axis: string; dataA: string; dataB: string;
  accent: string; action: string; text: string;
}
function readPalette(): Palette {
  const s = getComputedStyle(document.documentElement);
  const v = (n: string) => s.getPropertyValue(n).trim();
  return {
    surface: v('--color-surface'), axis: v('--viz-axis'),
    dataA: v('--viz-data-a'), dataB: v('--viz-data-b'),
    accent: v('--viz-accent'), action: v('--viz-action'), text: v('--color-muted'),
  };
}

const DOMAIN: [number, number, number, number] = [-5, 5, -5, 5];
const START = { w0: 2.5, w1: 1.5, b: 0, px: -1.5, py: 2.2 };
const MIN_NORM = 0.4; // keep ‖w‖ away from 0 so the boundary stays defined

export interface GeometryHandles {
  setBias(b: number): void;
  reset(): void;
  destroy(): void;
}

export interface MountOptions {
  onState?: (s: GeometryState) => void;
}

export function mountBoundaryGeometry(
  canvas: HTMLCanvasElement,
  opts: MountOptions = {},
): GeometryHandles {
  const maybeCtx = canvas.getContext('2d');
  if (!maybeCtx) throw new Error('2D canvas context unavailable');
  const ctx: CanvasRenderingContext2D = maybeCtx;

  const onState = opts.onState ?? (() => {});
  let palette = readPalette();
  let w0 = START.w0, w1 = START.w1, b = START.b;
  let px = START.px, py = START.py;
  let drag: 'w' | 'p' | null = null;

  const pad = 28;
  function toScreen(x: number, y: number): [number, number] {
    const [xmin, xmax, ymin, ymax] = DOMAIN;
    const sx = ((x - xmin) / (xmax - xmin)) * (canvas.width - 2 * pad) + pad;
    const sy = canvas.height - (((y - ymin) / (ymax - ymin)) * (canvas.height - 2 * pad) + pad);
    return [sx, sy];
  }
  function toWorld(sx: number, sy: number): [number, number] {
    const [xmin, xmax, ymin, ymax] = DOMAIN;
    const x = ((sx - pad) / (canvas.width - 2 * pad)) * (xmax - xmin) + xmin;
    const y = ((canvas.height - sy - pad) / (canvas.height - 2 * pad)) * (ymax - ymin) + ymin;
    return [x, y];
  }

  const zAt = (x: number, y: number) => w0 * x + w1 * y + b;

  function draw() {
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = palette.surface;
    ctx.fillRect(0, 0, width, height);

    // Half-planes: tint the z ≥ 0 side and the z < 0 side.
    const cell = 14;
    ctx.globalAlpha = 0.12;
    for (let sx = pad; sx < width - pad; sx += cell) {
      for (let sy = pad; sy < height - pad; sy += cell) {
        const [wx, wy] = toWorld(sx + cell / 2, sy + cell / 2);
        ctx.fillStyle = zAt(wx, wy) >= 0 ? palette.dataB : palette.dataA;
        ctx.fillRect(sx, sy, cell, cell);
      }
    }
    ctx.globalAlpha = 1;

    // Axes.
    ctx.strokeStyle = palette.axis;
    ctx.lineWidth = 1;
    const [ox, oy] = toScreen(0, 0);
    ctx.beginPath();
    ctx.moveTo(pad, oy); ctx.lineTo(width - pad, oy);
    ctx.moveTo(ox, pad); ctx.lineTo(ox, height - pad);
    ctx.stroke();

    drawBoundary();
    drawDistance();
    drawWeightVector();
    drawProbe();
  }

  function drawBoundary() {
    const [xmin, xmax, ymin, ymax] = DOMAIN;
    ctx.strokeStyle = palette.accent;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    if (Math.abs(w1) >= Math.abs(w0)) {
      const y1 = -(w0 * xmin + b) / w1;
      const y2 = -(w0 * xmax + b) / w1;
      const [ax, ay] = toScreen(xmin, y1);
      const [bx, by] = toScreen(xmax, y2);
      ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
    } else {
      const x1 = -(w1 * ymin + b) / w0;
      const x2 = -(w1 * ymax + b) / w0;
      const [ax, ay] = toScreen(x1, ymin);
      const [bx, by] = toScreen(x2, ymax);
      ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
    }
    ctx.stroke();
  }

  // The weight vector w, drawn as an arrow from the origin — the boundary's normal.
  function drawWeightVector() {
    const [ox, oy] = toScreen(0, 0);
    const [tx, ty] = toScreen(w0, w1);
    ctx.strokeStyle = palette.action;
    ctx.fillStyle = palette.action;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(ox, oy); ctx.lineTo(tx, ty);
    ctx.stroke();
    const ang = Math.atan2(ty - oy, tx - ox);
    const head = 11;
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(tx - head * Math.cos(ang - 0.4), ty - head * Math.sin(ang - 0.4));
    ctx.lineTo(tx - head * Math.cos(ang + 0.4), ty - head * Math.sin(ang + 0.4));
    ctx.closePath();
    ctx.fill();
    ctx.font = '600 13px ui-sans-serif, system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText('w', tx + 8, ty - 4);
  }

  // The probe point p and the perpendicular dropped to the boundary: its length
  // is |z| / ‖w‖, and it lands at foot = p − (z / ‖w‖²) w.
  function drawDistance() {
    const nn = w0 * w0 + w1 * w1;
    if (nn === 0) return;
    const z = zAt(px, py);
    const fx = px - (z / nn) * w0;
    const fy = py - (z / nn) * w1;
    const [sx, sy] = toScreen(px, py);
    const [gx, gy] = toScreen(fx, fy);
    ctx.strokeStyle = palette.action;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(sx, sy); ctx.lineTo(gx, gy);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function drawProbe() {
    const [sx, sy] = toScreen(px, py);
    const onPos = zAt(px, py) >= 0;
    ctx.beginPath();
    ctx.fillStyle = onPos ? palette.dataB : palette.dataA;
    ctx.arc(sx, sy, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = palette.text;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(sx, sy, 10, 0, Math.PI * 2);
    ctx.stroke();
    ctx.font = '600 13px ui-sans-serif, system-ui, sans-serif';
    ctx.fillStyle = palette.text;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('x', sx + 11, sy + 4);
  }

  function emit() {
    const normW = Math.hypot(w0, w1);
    const z = zAt(px, py);
    onState({ w0, w1, b, px, py, z, normW, dist: normW === 0 ? 0 : z / normW });
  }

  // ── pointer drag: grab whichever handle (w tip or probe) is nearer ──
  function pointerPos(ev: PointerEvent): [number, number] {
    const rect = canvas.getBoundingClientRect();
    const sx = ((ev.clientX - rect.left) / rect.width) * canvas.width;
    const sy = ((ev.clientY - rect.top) / rect.height) * canvas.height;
    return [sx, sy];
  }
  function dist2(ax: number, ay: number, bx: number, by: number) {
    return (ax - bx) ** 2 + (ay - by) ** 2;
  }
  function onDown(ev: PointerEvent) {
    const [sx, sy] = pointerPos(ev);
    const [wx, wy] = toScreen(w0, w1);
    const [ppx, ppy] = toScreen(px, py);
    const grab = 22 * 22;
    const dw = dist2(sx, sy, wx, wy);
    const dp = dist2(sx, sy, ppx, ppy);
    if (dw <= grab && dw <= dp) drag = 'w';
    else if (dp <= grab) drag = 'p';
    else return;
    canvas.setPointerCapture(ev.pointerId);
    ev.preventDefault();
  }
  function onMove(ev: PointerEvent) {
    if (!drag) return;
    const [sx, sy] = pointerPos(ev);
    const [wx, wy] = toWorld(sx, sy);
    const [xmin, xmax, ymin, ymax] = DOMAIN;
    const cx = Math.max(xmin, Math.min(xmax, wx));
    const cy = Math.max(ymin, Math.min(ymax, wy));
    if (drag === 'w') {
      // keep ‖w‖ above a floor so the boundary is always defined
      const n = Math.hypot(cx, cy);
      if (n < MIN_NORM) { w0 = MIN_NORM; w1 = 0; }
      else { w0 = cx; w1 = cy; }
    } else {
      px = cx; py = cy;
    }
    draw();
    emit();
  }
  function onUp(ev: PointerEvent) {
    if (!drag) return;
    drag = null;
    if (canvas.hasPointerCapture(ev.pointerId)) canvas.releasePointerCapture(ev.pointerId);
  }
  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerup', onUp);
  canvas.addEventListener('pointercancel', onUp);

  function onTheme() { palette = readPalette(); draw(); }
  window.addEventListener('themechange', onTheme);

  draw();
  emit();

  return {
    setBias(v: number) { b = v; draw(); emit(); },
    reset() {
      w0 = START.w0; w1 = START.w1; b = START.b; px = START.px; py = START.py;
      draw(); emit();
    },
    destroy() {
      canvas.removeEventListener('pointerdown', onDown);
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerup', onUp);
      canvas.removeEventListener('pointercancel', onUp);
      window.removeEventListener('themechange', onTheme);
    },
  };
}
