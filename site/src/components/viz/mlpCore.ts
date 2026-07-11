// Interactive MLP — vanilla TypeScript + Canvas, no framework.
// The playable companion to Module 01 (topics/01-mlp-backprop). It mirrors the
// same mechanism as the C and Python code: a 2 -> H -> 1 sigmoid network trained
// by backprop, drawn in TWO panels. The left panel is the input plane, where the
// decision boundary *bends* around two moons or the XOR corners. The right panel
// replots every data point at its hidden activations (a1_1, a1_2) — the learned
// coordinate system — where, once training succeeds, a single straight cut
// separates the classes (exact when hidden = 2; a 2-unit projection otherwise).
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
export type Mode = 'moons' | 'xor';

/** Tiny deterministic PRNG (mulberry32) so the scene is reproducible. */
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

/** Two interleaving half-moons — not linearly separable, but an MLP can bend a
 *  boundary between them. Built from sin/cos plus a little noise. */
function makeMoons(seed: number, perClass = 60, noise = 0.12): Pt[] {
  const rand = mulberry32(seed);
  const pts: Pt[] = [];
  for (let i = 0; i < perClass; i++) {
    const t = (Math.PI * i) / (perClass - 1);
    pts.push({ x: Math.cos(t) + noise * (2 * rand() - 1),
               y: Math.sin(t) + noise * (2 * rand() - 1), label: 0 });
    pts.push({ x: 1 - Math.cos(t) + noise * (2 * rand() - 1),
               y: 0.5 - Math.sin(t) + noise * (2 * rand() - 1), label: 1 });
  }
  return pts;
}

/** XOR — the four-corner problem from Module 00, now solvable with a hidden layer. */
function makeXor(): Pt[] {
  return [
    { x: 0, y: 0, label: 0 },
    { x: 1, y: 1, label: 0 },
    { x: 0, y: 1, label: 1 },
    { x: 1, y: 0, label: 1 },
  ];
}

// ── The network: 2 -> H -> 1, sigmoid hidden + sigmoid output ─────────────────

export interface Params { W1: number[][]; b1: number[]; W2: number[]; b2: number; }

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

function initParams(H: number, seed: number, scale = 1.0): Params {
  const rand = mulberry32(seed);
  const signed = () => scale * (2 * rand() - 1);
  const W1: number[][] = [];
  const b1: number[] = [];
  const W2: number[] = [];
  for (let j = 0; j < H; j++) { W1.push([signed(), signed()]); }
  for (let j = 0; j < H; j++) { b1.push(signed()); }
  for (let j = 0; j < H; j++) { W2.push(signed()); }
  return { W1, b1, W2, b2: signed() };
}

/** Forward pass: returns hidden activations and the output probability. */
function forward(p: Params, x: number, y: number): { a1: number[]; a2: number } {
  const H = p.W1.length;
  const a1 = new Array<number>(H);
  for (let j = 0; j < H; j++) a1[j] = sigmoid(p.b1[j] + p.W1[j][0] * x + p.W1[j][1] * y);
  let z2 = p.b2;
  for (let j = 0; j < H; j++) z2 += p.W2[j] * a1[j];
  return { a1, a2: sigmoid(z2) };
}

export function predict(p: Params, x: number, y: number): Label {
  return forward(p, x, y).a2 >= 0.5 ? 1 : 0;
}

export function accuracy(p: Params, pts: Pt[]): number {
  if (pts.length === 0) return 1;
  let ok = 0;
  for (const q of pts) if (predict(p, q.x, q.y) === q.label) ok++;
  return ok / pts.length;
}

function bceLoss(p: Params, pts: Pt[]): number {
  if (pts.length === 0) return 0;
  let total = 0;
  for (const q of pts) {
    const a2 = Math.min(1 - 1e-9, Math.max(1e-9, forward(p, q.x, q.y).a2));
    total += -(q.label * Math.log(a2) + (1 - q.label) * Math.log(1 - a2));
  }
  return total / pts.length;
}

/** One full-batch gradient-descent step — the exact chain rule from mlp.c/mlp.py. */
function sgdStep(p: Params, pts: Pt[], lr: number): void {
  const H = p.W1.length;
  const n = pts.length;
  if (n === 0) return;
  const dW1 = p.W1.map(() => [0, 0]);
  const db1 = new Array<number>(H).fill(0);
  const dW2 = new Array<number>(H).fill(0);
  let db2 = 0;
  for (const q of pts) {
    const { a1, a2 } = forward(p, q.x, q.y);
    const dz2 = a2 - q.label;                 // BCE through sigmoid
    for (let j = 0; j < H; j++) dW2[j] += dz2 * a1[j];
    db2 += dz2;
    for (let j = 0; j < H; j++) {
      const dz1 = dz2 * p.W2[j] * a1[j] * (1 - a1[j]); // through sigmoid'
      dW1[j][0] += dz1 * q.x;
      dW1[j][1] += dz1 * q.y;
      db1[j] += dz1;
    }
  }
  for (let j = 0; j < H; j++) {
    p.W1[j][0] -= lr * dW1[j][0] / n;
    p.W1[j][1] -= lr * dW1[j][1] / n;
    p.b1[j] -= lr * db1[j] / n;
    p.W2[j] -= lr * dW2[j] / n;
  }
  p.b2 -= lr * db2 / n;
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

type Domain = [number, number, number, number]; // [xmin, xmax, ymin, ymax]

/** Visible world domain of the input panel, per dataset. */
function domainFor(mode: Mode): Domain {
  return mode === 'moons' ? [-1.6, 2.6, -1.2, 1.7] : [-0.7, 1.7, -0.7, 1.7];
}

/** Hidden activations are sigmoid outputs, so the hidden panel is the unit
 *  square plus a small margin. */
const HIDDEN_DOMAIN: Domain = [-0.08, 1.08, -0.08, 1.08];

const PAD = 26;

/** World <-> screen mapping for one canvas and one domain. */
function mapper(c: HTMLCanvasElement, dom: Domain) {
  const [xmin, xmax, ymin, ymax] = dom;
  return {
    toScreen(x: number, y: number): [number, number] {
      const sx = ((x - xmin) / (xmax - xmin)) * (c.width - 2 * PAD) + PAD;
      const sy = c.height - (((y - ymin) / (ymax - ymin)) * (c.height - 2 * PAD) + PAD);
      return [sx, sy];
    },
    toWorld(sx: number, sy: number): [number, number] {
      const x = ((sx - PAD) / (c.width - 2 * PAD)) * (xmax - xmin) + xmin;
      const y = ((c.height - sy - PAD) / (c.height - 2 * PAD)) * (ymax - ymin) + ymin;
      return [x, y];
    },
  };
}

export interface Stats {
  loss: number; acc: number; epoch: number; hidden: number;
  playing: boolean; mode: Mode;
}

export interface MlpHandles {
  reset(): void;
  step(): void;
  toggle(): void;
  setMode(mode: Mode): void;
  setLearningRate(lr: number): void;
  setHidden(h: number): void;
  destroy(): void;
}

export interface MountOptions {
  seed?: number;
  /** Optional second canvas: the hidden-activation-space panel. */
  hiddenCanvas?: HTMLCanvasElement;
  onStats?: (s: Stats) => void;
}

const SEED = 7;
const STEPS_PER_TICK = 8;  // gradient steps folded into one visible "step"
const CELL = 12;           // decision-surface cell size, px

export function mountMlp(canvas: HTMLCanvasElement, opts: MountOptions = {}): MlpHandles {
  const maybeCtx = canvas.getContext('2d');
  if (!maybeCtx) throw new Error('2D canvas context unavailable');
  const ctx: CanvasRenderingContext2D = maybeCtx;
  const hCanvas = opts.hiddenCanvas ?? null;
  const hCtx = hCanvas ? hCanvas.getContext('2d') : null;

  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const seed = opts.seed ?? SEED;
  const onStats = opts.onStats ?? (() => {});

  let palette = readPalette();
  let mode: Mode = 'moons';
  let lr = 0.5;
  let hidden = 6;
  let pts: Pt[] = makeMoons(seed);
  let params: Params = initParams(hidden, seed);
  let epoch = 0;
  let playing = false;
  let raf = 0;

  function emit() {
    onStats({ loss: bceLoss(params, pts), acc: accuracy(params, pts),
              epoch, hidden, playing, mode });
  }

  /** Shared point renderer: dot per point, ring if misclassified. */
  function drawPoints(c: CanvasRenderingContext2D,
                      place: (q: Pt) => [number, number]) {
    for (const q of pts) {
      const [sx, sy] = place(q);
      c.beginPath();
      c.fillStyle = q.label === 0 ? palette.dataA : palette.dataB;
      c.arc(sx, sy, 5, 0, Math.PI * 2);
      c.fill();
      if (predict(params, q.x, q.y) !== q.label) {
        c.beginPath();
        c.strokeStyle = palette.text;
        c.lineWidth = 2;
        c.arc(sx, sy, 8, 0, Math.PI * 2);
        c.stroke();
      }
    }
  }

  function drawInput() {
    const { width, height } = canvas;
    const m = mapper(canvas, domainFor(mode));
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = palette.surface;
    ctx.fillRect(0, 0, width, height);

    // Decision surface: tint each cell by the output probability there — the
    // curved boundary the hidden layer is learning.
    for (let sx = PAD; sx < width - PAD; sx += CELL) {
      for (let sy = PAD; sy < height - PAD; sy += CELL) {
        const [wx, wy] = m.toWorld(sx + CELL / 2, sy + CELL / 2);
        const a2 = forward(params, wx, wy).a2;
        ctx.globalAlpha = 0.10 + 0.16 * Math.abs(a2 - 0.5) * 2; // stronger where confident
        ctx.fillStyle = a2 >= 0.5 ? palette.dataB : palette.dataA;
        ctx.fillRect(sx, sy, CELL, CELL);
      }
    }
    ctx.globalAlpha = 1;

    // Axes through the origin.
    ctx.strokeStyle = palette.axis;
    ctx.lineWidth = 1;
    const [ox, oy] = m.toScreen(0, 0);
    ctx.beginPath();
    ctx.moveTo(PAD, oy); ctx.lineTo(width - PAD, oy);
    ctx.moveTo(ox, PAD); ctx.lineTo(ox, height - PAD);
    ctx.stroke();

    drawPoints(ctx, (q) => m.toScreen(q.x, q.y));
  }

  /** The hidden-space panel: every point replotted at its hidden activations.
   *  With hidden = 1 or 2 the output unit's decision surface is exact here — a
   *  straight cut through the unit square. With hidden > 2 we can only show the
   *  first two coordinates of an H-dimensional space, so the surface is hidden
   *  and a hint says so. */
  function drawHidden() {
    if (!hCanvas || !hCtx) return;
    const c = hCtx;
    const { width, height } = hCanvas;
    const m = mapper(hCanvas, HIDDEN_DOMAIN);
    c.clearRect(0, 0, width, height);
    c.fillStyle = palette.surface;
    c.fillRect(0, 0, width, height);

    // Output unit's decision surface over the hidden square — only drawable
    // when the hidden space genuinely is <= 2-dimensional.
    if (hidden <= 2) {
      for (let sx = PAD; sx < width - PAD; sx += CELL) {
        for (let sy = PAD; sy < height - PAD; sy += CELL) {
          const [ha, hb] = m.toWorld(sx + CELL / 2, sy + CELL / 2);
          const z2 = hidden === 2
            ? params.b2 + params.W2[0] * ha + params.W2[1] * hb
            : params.b2 + params.W2[0] * ha;   // 1-D hidden space: a2 ignores hb
          const a2 = sigmoid(z2);
          c.globalAlpha = 0.10 + 0.16 * Math.abs(a2 - 0.5) * 2;
          c.fillStyle = a2 >= 0.5 ? palette.dataB : palette.dataA;
          c.fillRect(sx, sy, CELL, CELL);
        }
      }
      c.globalAlpha = 1;
    }

    // Frame the unit square (activations live in (0,1) x (0,1)).
    const [x0, y0] = m.toScreen(0, 0);
    const [x1, y1] = m.toScreen(1, 1);
    c.strokeStyle = palette.axis;
    c.lineWidth = 1;
    c.strokeRect(x1 < x0 ? x1 : x0, y1 < y0 ? y1 : y0, Math.abs(x1 - x0), Math.abs(y1 - y0));

    // Tick labels at 0 and 1 on both axes.
    c.fillStyle = palette.text;
    c.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    c.textAlign = 'center';
    c.fillText('0', x0, y0 + 14);
    c.fillText('1', x1, y0 + 14);
    c.textAlign = 'right';
    c.fillText('0', x0 - 6, y0 + 4);
    c.fillText('1', x0 - 6, y1 + 4);
    c.textAlign = 'left';

    // Points at their hidden coordinates. With one hidden unit the space is a
    // line, so everything sits on the axis midline.
    drawPoints(c, (q) => {
      const { a1 } = forward(params, q.x, q.y);
      return m.toScreen(a1[0], hidden >= 2 ? a1[1] : 0.5);
    });

    // Honest labelling when we can only draw a projection.
    if (hidden > 2) {
      c.fillStyle = palette.text;
      c.font = '11px ui-sans-serif, system-ui, sans-serif';
      c.textAlign = 'center';
      c.fillText(`showing units 1–2 of ${hidden} — set hidden = 2 to see the output cut`,
                 width / 2, height - 8);
      c.textAlign = 'left';
    } else if (hidden === 1) {
      c.fillStyle = palette.text;
      c.font = '11px ui-sans-serif, system-ui, sans-serif';
      c.textAlign = 'center';
      c.fillText('1 hidden unit — the hidden space is a line', width / 2, height - 8);
      c.textAlign = 'left';
    }
  }

  function draw() {
    drawInput();
    drawHidden();
  }

  function stepOnce() {
    for (let k = 0; k < STEPS_PER_TICK; k++) sgdStep(params, pts, lr);
    epoch += STEPS_PER_TICK;
    draw();
    emit();
  }

  let acc_t = 0;
  let prev = 0;
  const stepMs = 90;
  function loop(now: number) {
    if (!playing) return;
    if (prev === 0) prev = now;
    acc_t += now - prev;
    prev = now;
    while (acc_t >= stepMs) { stepOnce(); acc_t -= stepMs; }
    raf = requestAnimationFrame(loop);
  }
  function play() {
    if (reducedMotion) return;
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
    params = initParams(hidden, seed);
    epoch = 0;
    pts = mode === 'moons' ? makeMoons(seed) : makeXor();
    draw();
    emit();
  }

  function onTheme() { palette = readPalette(); draw(); }
  window.addEventListener('themechange', onTheme);

  draw();
  emit();

  return {
    reset: resetScene,
    step() { pause(); stepOnce(); },
    toggle() { playing ? pause() : play(); },
    setMode(m: Mode) { mode = m; resetScene(); },
    setLearningRate(v: number) { lr = v; emit(); },
    setHidden(h: number) { hidden = h; resetScene(); },
    destroy() {
      pause();
      window.removeEventListener('themechange', onTheme);
    },
  };
}
