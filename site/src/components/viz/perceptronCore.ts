// Interactive perceptron — vanilla TypeScript + Canvas, no framework.
// The playable companion to Module 00 (topics/00-perceptron). It mirrors the
// same mechanism as the C and Python code: z = w0*x + w1*y + b, predict 1 when
// z >= 0, and learn by Rosenblatt's rule w <- w + lr*(y - yhat)*x.
//
// Follows the widget conventions in docs/conventions/viz-style.md:
//   • semantic colors read live from the theme (CSS variables), never hardcoded
//   • repaint on `themechange` (dispatched by BaseLayout's theme toggle)
//   • respect prefers-reduced-motion (no autoplay; stepping still works)
//   • deterministic seed + a reset control
//
// The .astro component stays thin: it grabs the DOM controls and wires them to
// the handles returned here.

// ── Model & data ────────────────────────────────────────────────────────────

export type Label = 0 | 1;
export interface Pt { x: number; y: number; label: Label; }
export interface Model { w0: number; w1: number; b: number; }
export type Mode = 'blobs' | 'xor';

/** Tiny deterministic PRNG (mulberry32) so the blob scene is reproducible. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Two linearly-separable 2D blobs — the case the perceptron can solve.
 *
 * Both centers sit in the positive quadrant, on the *same* side of the origin,
 * and this is deliberate. The model starts at w=b=0, i.e. a boundary through
 * the origin. If the two blobs straddled the origin (as symmetric ±centers
 * would), that starting line would already separate them and the perceptron
 * would be "done" after a single update — nothing to watch. With both blobs
 * off to one side, the ideal boundary is roughly x + y = 3, so the bias must
 * accumulate over many updates and you actually see the line swing into place.
 * The spread is kept tight relative to the center gap so the classes stay
 * cleanly separable under the fixed seed (converges to 100% in ~12 epochs).
 */
function makeBlobs(seed: number): Pt[] {
  const rand = mulberry32(seed);
  const centers: Array<[number, number]> = [[0.5, 0.5], [2.5, 2.5]];
  const spread = 1.2;
  const pts: Pt[] = [];
  for (let c = 0 as Label; c <= 1; c = (c + 1) as Label) {
    for (let k = 0; k < 14; k++) {
      pts.push({
        x: centers[c][0] + spread * (2 * rand() - 1),
        y: centers[c][1] + spread * (2 * rand() - 1),
        label: c,
      });
    }
  }
  return pts;
}

/** XOR — NOT linearly separable, so a single perceptron can never solve it. */
function makeXor(): Pt[] {
  return [
    { x: 0, y: 0, label: 0 },
    { x: 1, y: 1, label: 0 },
    { x: 0, y: 1, label: 1 },
    { x: 1, y: 0, label: 1 },
  ];
}

export function preactivation(m: Model, x: number, y: number): number {
  return m.w0 * x + m.w1 * y + m.b; // z = w0*x + w1*y + b
}
export function predict(m: Model, x: number, y: number): Label {
  return preactivation(m, x, y) >= 0 ? 1 : 0;
}
export function accuracy(m: Model, pts: Pt[]): number {
  if (pts.length === 0) return 1;
  let ok = 0;
  for (const p of pts) if (predict(m, p.x, p.y) === p.label) ok++;
  return ok / pts.length;
}
/** One perceptron update from a single point. Returns true if it was wrong. */
function learnFrom(m: Model, p: Pt, lr: number): boolean {
  const yhat = predict(m, p.x, p.y);
  const err = p.label - yhat; // -1, 0, or +1
  m.w0 += lr * err * p.x;
  m.w1 += lr * err * p.y;
  m.b += lr * err;
  return err !== 0;
}

// ── Rendering + interaction ──────────────────────────────────────────────────

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
    accent: v('--viz-accent'), action: v('--viz-action'),
    text: v('--color-muted'),
  };
}

/** Visible world domain per mode: [xmin, xmax, ymin, ymax]. */
function domainFor(mode: Mode): [number, number, number, number] {
  return mode === 'blobs' ? [-5, 5, -5, 5] : [-0.7, 1.7, -0.7, 1.7];
}

export interface Stats {
  w0: number; w1: number; b: number;
  acc: number; epoch: number; updates: number; playing: boolean; mode: Mode;
}

export interface PerceptronHandles {
  reset(): void;
  step(): void;
  toggle(): void;
  setMode(mode: Mode): void;
  setLearningRate(lr: number): void;
  destroy(): void;
}

export interface MountOptions {
  seed?: number;
  onStats?: (s: Stats) => void;
}

const BLOB_SEED = 7;

export function mountPerceptron(
  canvas: HTMLCanvasElement,
  opts: MountOptions = {},
): PerceptronHandles {
  const maybeCtx = canvas.getContext('2d');
  if (!maybeCtx) throw new Error('2D canvas context unavailable');
  const ctx: CanvasRenderingContext2D = maybeCtx;

  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const seed = opts.seed ?? BLOB_SEED;
  const onStats = opts.onStats ?? (() => {});

  let palette = readPalette();
  let mode: Mode = 'blobs';
  let lr = 0.05;
  let pts: Pt[] = makeBlobs(seed);
  let model: Model = { w0: 0, w1: 0, b: 0 };
  let cursor = 0;      // index of the next point the learning rule will visit
  let epoch = 0;       // full passes over the dataset
  let updates = 0;     // number of actual weight changes
  let lastWrong = -1;  // index highlighted as the most recent mistake
  let playing = false;
  let raf = 0;
  let acc = 0;

  // ── coordinate transforms (world <-> canvas pixels) ──
  const pad = 26;
  function toScreen(x: number, y: number): [number, number] {
    const [xmin, xmax, ymin, ymax] = domainFor(mode);
    const sx = ((x - xmin) / (xmax - xmin)) * (canvas.width - 2 * pad) + pad;
    const sy = canvas.height - (((y - ymin) / (ymax - ymin)) * (canvas.height - 2 * pad) + pad);
    return [sx, sy];
  }
  function toWorld(sx: number, sy: number): [number, number] {
    const [xmin, xmax, ymin, ymax] = domainFor(mode);
    const x = ((sx - pad) / (canvas.width - 2 * pad)) * (xmax - xmin) + xmin;
    const y = ((canvas.height - sy - pad) / (canvas.height - 2 * pad)) * (ymax - ymin) + ymin;
    return [x, y];
  }

  function emit() {
    acc = accuracy(model, pts);
    onStats({ w0: model.w0, w1: model.w1, b: model.b, acc, epoch, updates, playing, mode });
  }

  // ── drawing ──
  function draw() {
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = palette.surface;
    ctx.fillRect(0, 0, width, height);

    // Decision regions: tint each cell by the model's prediction there. This is
    // the "surface" the perceptron is carving out of the plane.
    const cell = 14;
    ctx.globalAlpha = 0.14;
    for (let sx = pad; sx < width - pad; sx += cell) {
      for (let sy = pad; sy < height - pad; sy += cell) {
        const [wx, wy] = toWorld(sx + cell / 2, sy + cell / 2);
        ctx.fillStyle = predict(model, wx, wy) === 1 ? palette.dataB : palette.dataA;
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

    // Decision boundary: the line where z = 0, i.e. w0*x + w1*y + b = 0.
    drawBoundary();

    // Data points, colored by true class; misclassified ones get a ring, and
    // the most recent mistake is highlighted in the "action" color.
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      const [sx, sy] = toScreen(p.x, p.y);
      const correct = predict(model, p.x, p.y) === p.label;
      ctx.beginPath();
      ctx.fillStyle = p.label === 0 ? palette.dataA : palette.dataB;
      ctx.arc(sx, sy, 6, 0, Math.PI * 2);
      ctx.fill();
      if (!correct) {
        ctx.beginPath();
        ctx.strokeStyle = palette.text;
        ctx.lineWidth = 2;
        ctx.arc(sx, sy, 9, 0, Math.PI * 2);
        ctx.stroke();
      }
      if (i === lastWrong) {
        ctx.beginPath();
        ctx.strokeStyle = palette.action;
        ctx.lineWidth = 3;
        ctx.arc(sx, sy, 11, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }

  function drawBoundary() {
    const { w0, w1, b } = model;
    if (w0 === 0 && w1 === 0) return; // no line defined yet
    const [xmin, xmax, ymin, ymax] = domainFor(mode);
    ctx.strokeStyle = palette.accent;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    if (Math.abs(w1) >= Math.abs(w0)) {
      // solve y from x across the visible x-range
      const y1 = -(w0 * xmin + b) / w1;
      const y2 = -(w0 * xmax + b) / w1;
      const [ax, ay] = toScreen(xmin, y1);
      const [bx, by] = toScreen(xmax, y2);
      ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
    } else {
      // near-vertical: solve x from y across the visible y-range
      const x1 = -(w1 * ymin + b) / w0;
      const x2 = -(w1 * ymax + b) / w0;
      const [ax, ay] = toScreen(x1, ymin);
      const [bx, by] = toScreen(x2, ymax);
      ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
    }
    ctx.stroke();
  }

  // ── learning loop ──
  function stepOnce() {
    if (pts.length === 0) return;
    const idx = cursor % pts.length;
    const wrong = learnFrom(model, pts[idx], lr);
    lastWrong = wrong ? idx : -1;
    if (wrong) updates++;
    cursor++;
    if (cursor % pts.length === 0) epoch++;
    draw();
    emit();
  }

  // Autoplay is throttled so a human can watch each update land.
  let acc_t = 0;
  let prev = 0;
  const stepMs = 140;
  function loop(now: number) {
    if (!playing) return;
    if (prev === 0) prev = now;
    acc_t += now - prev;
    prev = now;
    while (acc_t >= stepMs) { stepOnce(); acc_t -= stepMs; }
    raf = requestAnimationFrame(loop);
  }
  function play() {
    if (reducedMotion) return; // honor reduced motion: no autoplay
    playing = true; prev = 0; acc_t = 0;
    raf = requestAnimationFrame(loop);
    emit();
  }
  function pause() {
    playing = false;
    cancelAnimationFrame(raf);
    emit();
  }

  function resetScene() {
    pause();
    model = { w0: 0, w1: 0, b: 0 };
    cursor = 0; epoch = 0; updates = 0; lastWrong = -1;
    pts = mode === 'blobs' ? makeBlobs(seed) : makeXor();
    draw();
    emit();
  }

  // ── interaction: click to add a point of the class nearer the click ──
  function onClick(ev: MouseEvent) {
    const rect = canvas.getBoundingClientRect();
    const sx = ((ev.clientX - rect.left) / rect.width) * canvas.width;
    const sy = ((ev.clientY - rect.top) / rect.height) * canvas.height;
    const [wx, wy] = toWorld(sx, sy);
    // Shift-click adds class B (orange); plain click adds class A (blue).
    const label: Label = ev.shiftKey ? 1 : 0;
    pts.push({ x: wx, y: wy, label });
    draw();
    emit();
  }
  canvas.addEventListener('click', onClick);

  function onTheme() { palette = readPalette(); draw(); }
  window.addEventListener('themechange', onTheme);

  // initial paint
  draw();
  emit();

  return {
    reset: resetScene,
    step() { pause(); stepOnce(); },
    toggle() { playing ? pause() : play(); },
    setMode(m: Mode) { mode = m; resetScene(); },
    setLearningRate(v: number) { lr = v; emit(); },
    destroy() {
      pause();
      canvas.removeEventListener('click', onClick);
      window.removeEventListener('themechange', onTheme);
    },
  };
}
