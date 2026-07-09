// Interactive McCulloch–Pitts logic gates — vanilla TypeScript + Canvas.
// Companion to Module 00's "from brain cell to arithmetic" section. It shows the
// 1943 insight: a single thresholded unit z = w0*x0 + w1*x1 + b, firing when
// z >= 0, already computes AND, OR, and NOT — if you set the weights by hand.
// Toggle the gate and watch one straight line split its truth table. The point
// it can't make is exactly XOR, which sets up the rest of the chapter.
//
// Follows docs/conventions/viz-style.md, same lifecycle as perceptronCore.ts:
//   • semantic colors read live from the theme (CSS variables), never hardcoded
//   • repaint on `themechange`
//   • deterministic (no RNG) + a reset control
//   • labels use the math-notation symbols (w, b, z)

export type Gate = 'and' | 'or' | 'not';

export interface Model { w0: number; w1: number; b: number; }

/** Hand-set weights + bias that make one thresholded unit behave like each gate.
 *  Firing test is z = w0*x0 + w1*x1 + b >= 0, matching the site's z = w·x + b
 *  convention (the classic McCulloch–Pitts threshold is θ = −b). */
const GATES: Record<Gate, Model> = {
  and: { w0: 1, w1: 1, b: -1.5 }, // fires only at (1,1):  x0+x1 ≥ 1.5
  or: { w0: 1, w1: 1, b: -0.5 }, // fires unless (0,0):  x0+x1 ≥ 0.5
  not: { w0: -1, w1: 0, b: 0.5 }, // one input: fires when x0 = 0  (−x0 ≥ −0.5)
};

export interface GateRow { x0: number; x1: number | null; z: number; out: 0 | 1; }

export interface GateState {
  gate: Gate;
  w0: number; w1: number; b: number;
  rows: GateRow[];
}

function preactivation(m: Model, x0: number, x1: number): number {
  return m.w0 * x0 + m.w1 * x1 + m.b;
}
function fires(m: Model, x0: number, x1: number): 0 | 1 {
  return preactivation(m, x0, x1) >= 0 ? 1 : 0;
}

/** The points that make up each gate's truth table (NOT is a single input). */
function inputsFor(gate: Gate): Array<[number, number | null]> {
  if (gate === 'not') return [[0, null], [1, null]];
  return [[0, 0], [0, 1], [1, 0], [1, 1]];
}

function truthTable(gate: Gate, m: Model): GateRow[] {
  return inputsFor(gate).map(([x0, x1]) => {
    const y = x1 ?? 0.5; // NOT ignores x1; plot/evaluate it at the mid-line
    return { x0, x1, z: preactivation(m, x0, y), out: fires(m, x0, y) };
  });
}

// ── rendering ────────────────────────────────────────────────────────────────

interface Palette {
  surface: string; axis: string; dataA: string; dataB: string;
  accent: string; text: string;
}
function readPalette(): Palette {
  const s = getComputedStyle(document.documentElement);
  const v = (n: string) => s.getPropertyValue(n).trim();
  return {
    surface: v('--color-surface'), axis: v('--viz-axis'),
    dataA: v('--viz-data-a'), dataB: v('--viz-data-b'),
    accent: v('--viz-accent'), text: v('--color-muted'),
  };
}

const DOMAIN: [number, number, number, number] = [-0.45, 1.45, -0.45, 1.45];

export interface GateHandles {
  setGate(gate: Gate): void;
  reset(): void;
  destroy(): void;
}

export interface MountOptions {
  onState?: (s: GateState) => void;
}

export function mountLogicGate(
  canvas: HTMLCanvasElement,
  opts: MountOptions = {},
): GateHandles {
  const maybeCtx = canvas.getContext('2d');
  if (!maybeCtx) throw new Error('2D canvas context unavailable');
  const ctx: CanvasRenderingContext2D = maybeCtx;

  const onState = opts.onState ?? (() => {});
  let palette = readPalette();
  let gate: Gate = 'and';
  let model: Model = { ...GATES[gate] };

  const pad = 30;
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

  function draw() {
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = palette.surface;
    ctx.fillRect(0, 0, width, height);

    // Decision regions: tint each side by what the unit fires there.
    const cell = 14;
    ctx.globalAlpha = 0.13;
    for (let sx = pad; sx < width - pad; sx += cell) {
      for (let sy = pad; sy < height - pad; sy += cell) {
        const [wx, wy] = toWorld(sx + cell / 2, sy + cell / 2);
        ctx.fillStyle = fires(model, wx, wy) === 1 ? palette.dataB : palette.dataA;
        ctx.fillRect(sx, sy, cell, cell);
      }
    }
    ctx.globalAlpha = 1;

    // Axes through the origin.
    ctx.strokeStyle = palette.axis;
    ctx.lineWidth = 1;
    const [ox, oy] = toScreen(0, 0);
    ctx.beginPath();
    ctx.moveTo(pad, oy); ctx.lineTo(width - pad, oy);
    ctx.moveTo(ox, pad); ctx.lineTo(ox, height - pad);
    ctx.stroke();

    // The separating line z = 0.
    drawBoundary();

    // Truth-table points, colored + labelled by the unit's output.
    ctx.font = '600 13px ui-sans-serif, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const row of truthTable(gate, model)) {
      const y = row.x1 ?? 0.5;
      const [sx, sy] = toScreen(row.x0, y);
      ctx.beginPath();
      ctx.fillStyle = row.out === 1 ? palette.dataB : palette.dataA;
      ctx.arc(sx, sy, 11, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.fillText(String(row.out), sx, sy + 0.5);
    }
  }

  function drawBoundary() {
    const { w0, w1, b } = model;
    if (w0 === 0 && w1 === 0) return;
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
      // near-vertical (e.g. NOT): solve x from y across the visible y-range
      const x1 = -(w1 * ymin + b) / w0;
      const x2 = -(w1 * ymax + b) / w0;
      const [ax, ay] = toScreen(x1, ymin);
      const [bx, by] = toScreen(x2, ymax);
      ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
    }
    ctx.stroke();
  }

  function emit() {
    onState({ gate, w0: model.w0, w1: model.w1, b: model.b, rows: truthTable(gate, model) });
  }

  function apply(g: Gate) {
    gate = g;
    model = { ...GATES[g] };
    draw();
    emit();
  }

  function onTheme() { palette = readPalette(); draw(); }
  window.addEventListener('themechange', onTheme);

  draw();
  emit();

  return {
    setGate: apply,
    reset() { apply('and'); },
    destroy() { window.removeEventListener('themechange', onTheme); },
  };
}
