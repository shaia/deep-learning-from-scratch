// Module 04's interactive: dropout versus memorization, live.
//
// A wide 2 -> H -> H -> 2 ReLU network trains (full batch, Adam) on a SMALL,
// NOISY two-moons sample -- a few dozen points, some with flipped labels.
// That is AlexNet's predicament in miniature: far more parameters than
// examples. With dropout off, the network carves a wiggly boundary around
// every training point, mislabeled ones included, and training accuracy
// races to 100% while accuracy on fresh points from the same distribution
// stalls. Turn dropout up and every step trains a different random "thinned"
// sub-network; no unit can rely on a partner being there, the boundary
// smooths out, and the gap between train and test accuracy closes.
//
// Three panels: the input plane (decision regions + the training points;
// mislabeled ones ringed), the thinned network of the most recent step (each
// hidden unit lit if it survived, hollow if dropped), and the train/test
// accuracy curves. The mechanism is the library's: inverted dropout --
// survivors scaled by 1/(1-p) in training, the full net at evaluation -- the
// same arithmetic as lib/python/nanograd/layers.py's Dropout.
//
// Widget conventions (docs/conventions/viz-style.md): theme colors from CSS
// variables, repaint on `themechange`, respect prefers-reduced-motion (no
// autoplay; Step still works), deterministic seed + a reset control.

// ── Tunables (chosen with a NumPy prototype of this exact setup) ─────────────
const SEED = 11;
const N_TRAIN = 40;          // the small training set
const N_TEST = 600;          // fresh points from the same distribution
const NOISE = 0.3;           // Gaussian jitter on the moons
const LABEL_FLIP = 0.1;      // fraction of training labels flipped
const HIDDEN = 64;           // units per hidden layer: wide, on purpose
const LR = 0.01;             // Adam
const MAX_STEPS = 3000;      // the run stops here (curves stay on screen)
const STEPS_PER_TICK = 10;
const TICK_MS = 60;
const CELL = 8;              // decision-surface cell size, px
export const DROP_CHOICES = [0, 0.25, 0.5, 0.75] as const;

// ── Data ─────────────────────────────────────────────────────────────────────

type Label = 0 | 1;
interface Pt { x: number; y: number; label: Label; flipped: boolean }

/** Tiny deterministic PRNG (mulberry32), as in the other widgets. */
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

/** Standard normal from two uniforms (Box-Muller). */
function gauss(rand: () => number): number {
  let u = rand();
  while (u <= 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

/** n points on two interleaving half-moons, alternating class, with jitter. */
function makeMoons(n: number, rand: () => number): Pt[] {
  const pts: Pt[] = [];
  for (let i = 0; i < n; i++) {
    const label = (i % 2) as Label;
    const t = Math.PI * rand();
    const x = label === 0 ? Math.cos(t) : 1 - Math.cos(t);
    const y = label === 0 ? Math.sin(t) : 0.5 - Math.sin(t);
    pts.push({ x: x + NOISE * gauss(rand), y: y + NOISE * gauss(rand), label, flipped: false });
  }
  return pts;
}

/** Centre and scale the raw moon coordinates into roughly [-1.5, 1.5]. */
const nx = (x: number) => (x - 0.5) / 1.2;
const ny = (y: number) => (y - 0.25) / 0.8;

// ── The network: 2 -> H -> H -> 2, ReLU, softmax cross-entropy, Adam ─────────

interface Layer { W: Float64Array; b: Float64Array; nIn: number; nOut: number }

function makeLayer(nIn: number, nOut: number, rand: () => number): Layer {
  const W = new Float64Array(nIn * nOut);
  const std = Math.sqrt(2 / nIn); // He init, as nanograd's he_normal
  for (let i = 0; i < W.length; i++) W[i] = std * gauss(rand);
  return { W, b: new Float64Array(nOut), nIn, nOut };
}

/** Y = X W + b for a batch (row-major X: m x nIn). */
function linear(L: Layer, X: Float64Array, m: number): Float64Array {
  const Y = new Float64Array(m * L.nOut);
  for (let i = 0; i < m; i++)
    for (let o = 0; o < L.nOut; o++) {
      let s = L.b[o];
      for (let k = 0; k < L.nIn; k++) s += X[i * L.nIn + k] * L.W[k * L.nOut + o];
      Y[i * L.nOut + o] = s;
    }
  return Y;
}

class Net {
  layers: Layer[];
  // Adam moments, one pair per parameter array: [W1, b1, W2, b2, W3, b3].
  m: Float64Array[];
  v: Float64Array[];
  t = 0;
  /** The most recent training step's dropout masks (1 = kept), per hidden layer. */
  lastMask: Uint8Array[] = [];

  constructor(rand: () => number) {
    this.layers = [makeLayer(2, HIDDEN, rand), makeLayer(HIDDEN, HIDDEN, rand), makeLayer(HIDDEN, 2, rand)];
    const arrays = this.layers.flatMap((L) => [L.W, L.b]);
    this.m = arrays.map((a) => new Float64Array(a.length));
    this.v = arrays.map((a) => new Float64Array(a.length));
    this.lastMask = [new Uint8Array(HIDDEN).fill(1), new Uint8Array(HIDDEN).fill(1)];
  }

  /** Evaluation-mode logits for one point: the full network, no dropout. */
  logits(x: number, y: number): [number, number] {
    let a: Float64Array = Float64Array.of(nx(x), ny(y));
    for (let l = 0; l < 3; l++) {
      a = linear(this.layers[l], a, 1);
      if (l < 2) for (let j = 0; j < a.length; j++) a[j] = Math.max(0, a[j]);
    }
    return [a[0], a[1]];
  }

  predict(x: number, y: number): Label {
    const [z0, z1] = this.logits(x, y);
    return z1 > z0 ? 1 : 0;
  }

  /** One full-batch Adam step with inverted dropout on both hidden layers --
   * forward, softmax cross-entropy, and the chain rule back, by hand. */
  step(pts: Pt[], p: number, rand: () => number): void {
    const m = pts.length;
    const keep = 1 - p;
    const X = new Float64Array(m * 2);
    pts.forEach((q, i) => { X[2 * i] = nx(q.x); X[2 * i + 1] = ny(q.y); });

    // Forward, caching each layer's input, pre-activation, and mask.
    const inputs: Float64Array[] = [];
    const pres: Float64Array[] = [];
    const masks: Float64Array[] = [];
    let a: Float64Array = X;
    for (let l = 0; l < 3; l++) {
      inputs.push(a);
      const z = linear(this.layers[l], a, m);
      if (l === 2) { a = z; break; }
      pres.push(z);
      // ReLU, then dropout: one mask per (point, unit), survivors x 1/keep.
      const mask = new Float64Array(m * HIDDEN);
      const out = new Float64Array(m * HIDDEN);
      for (let j = 0; j < out.length; j++) {
        mask[j] = p === 0 || rand() < keep ? 1 / keep : 0;
        out[j] = Math.max(0, z[j]) * mask[j];
      }
      masks.push(mask);
      a = out;
    }
    // The diagram shows the thinned network the FIRST training point saw.
    this.lastMask = masks.map((mk) => Uint8Array.from(mk.subarray(0, HIDDEN), (v) => (v > 0 ? 1 : 0)));

    // Softmax cross-entropy: dL/dlogits = (P - onehot(y)) / m.
    let d = new Float64Array(m * 2);
    for (let i = 0; i < m; i++) {
      const z0 = a[2 * i], z1 = a[2 * i + 1];
      const mx = Math.max(z0, z1);
      const e0 = Math.exp(z0 - mx), e1 = Math.exp(z1 - mx);
      const p1 = e1 / (e0 + e1);
      d[2 * i] = ((1 - p1) - (pts[i].label === 0 ? 1 : 0)) / m;
      d[2 * i + 1] = (p1 - (pts[i].label === 1 ? 1 : 0)) / m;
    }

    // Backward, layer by layer: dW = X^T d, db = sum d, dX = d W^T; then
    // through dropout (x mask) and ReLU (z > 0) on the way down.
    const grads: Float64Array[] = [];
    for (let l = 2; l >= 0; l--) {
      const L = this.layers[l];
      const Xin = inputs[l];
      const dW = new Float64Array(L.W.length);
      const db = new Float64Array(L.nOut);
      for (let i = 0; i < m; i++)
        for (let o = 0; o < L.nOut; o++) {
          const g = d[i * L.nOut + o];
          if (g === 0) continue;
          db[o] += g;
          for (let k = 0; k < L.nIn; k++) dW[k * L.nOut + o] += Xin[i * L.nIn + k] * g;
        }
      grads.unshift(dW, db);
      if (l === 0) break;
      const dX = new Float64Array(m * L.nIn);
      for (let i = 0; i < m; i++)
        for (let k = 0; k < L.nIn; k++) {
          let s = 0;
          for (let o = 0; o < L.nOut; o++) s += d[i * L.nOut + o] * L.W[k * L.nOut + o];
          const j = i * L.nIn + k;
          // A dropped unit, or one ReLU switched off, passes no gradient back.
          dX[j] = pres[l - 1][j] > 0 ? s * masks[l - 1][j] : 0;
        }
      d = dX;
    }

    // Adam (Kingma & Ba), the Module-02 optimizer.
    this.t += 1;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    const params = this.layers.flatMap((L) => [L.W, L.b]);
    params.forEach((P, k) => {
      const g = grads[k], mm = this.m[k], vv = this.v[k];
      for (let j = 0; j < P.length; j++) {
        mm[j] = b1 * mm[j] + (1 - b1) * g[j];
        vv[j] = b2 * vv[j] + (1 - b2) * g[j] * g[j];
        P[j] -= (LR * (mm[j] / c1)) / (Math.sqrt(vv[j] / c2) + eps);
      }
    });
  }
}

function accuracy(net: Net, pts: Pt[]): number {
  let ok = 0;
  for (const q of pts) if (net.predict(q.x, q.y) === q.label) ok++;
  return ok / pts.length;
}

// ── Rendering + interaction ──────────────────────────────────────────────────

interface Palette {
  surface: string; axis: string; text: string; muted: string;
  dataA: string; dataB: string; accent: string; action: string;
}
function readPalette(): Palette {
  const s = getComputedStyle(document.documentElement);
  const v = (n: string) => s.getPropertyValue(n).trim();
  return {
    surface: v('--color-surface'), axis: v('--viz-axis'), text: v('--color-text'),
    muted: v('--color-muted'), dataA: v('--viz-data-a'), dataB: v('--viz-data-b'),
    accent: v('--viz-accent'), action: v('--viz-action'),
  };
}

const DOMAIN = { xmin: -1.6, xmax: 2.6, ymin: -1.4, ymax: 1.9 };
const PAD = 14;

export interface DropoutStats {
  step: number;
  trainAcc: number;
  testAcc: number;
  p: number;
  playing: boolean;
  done: boolean;
}

export interface DropoutHandles {
  toggle(): void;
  step(): void;
  reset(): void;
  setDrop(p: number): void;
  destroy(): void;
  reducedMotion: boolean;
}

export function mountDropout(
  plane: HTMLCanvasElement,
  side: HTMLCanvasElement,
  opts: { onStats?: (s: DropoutStats) => void } = {},
): DropoutHandles {
  const c1 = plane.getContext('2d');
  const c2 = side.getContext('2d');
  if (!c1 || !c2) throw new Error('2D canvas context unavailable');
  const ctx: CanvasRenderingContext2D = c1;
  const sctx: CanvasRenderingContext2D = c2;

  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let palette = readPalette();
  let p = 0.5;
  let playing = false;
  let raf = 0;

  // Data is fixed by the seed; the network restarts from the same init.
  const dataRand = mulberry32(SEED);
  const train = makeMoons(N_TRAIN, dataRand);
  for (const q of train)
    if (dataRand() < LABEL_FLIP) { q.label = (1 - q.label) as Label; q.flipped = true; }
  const test = makeMoons(N_TEST, dataRand);

  let net: Net;
  let dropRand: () => number;
  let steps = 0;
  let history: { train: number; test: number }[] = [];

  function restart() {
    net = new Net(mulberry32(SEED + 1));
    dropRand = mulberry32(SEED + 2);
    steps = 0;
    history = [{ train: accuracy(net, train), test: accuracy(net, test) }];
  }

  const toScreen = (x: number, y: number): [number, number] => [
    PAD + ((x - DOMAIN.xmin) / (DOMAIN.xmax - DOMAIN.xmin)) * (plane.width - 2 * PAD),
    plane.height - PAD - ((y - DOMAIN.ymin) / (DOMAIN.ymax - DOMAIN.ymin)) * (plane.height - 2 * PAD),
  ];
  const toWorld = (sx: number, sy: number): [number, number] => [
    DOMAIN.xmin + ((sx - PAD) / (plane.width - 2 * PAD)) * (DOMAIN.xmax - DOMAIN.xmin),
    DOMAIN.ymin + ((plane.height - PAD - sy) / (plane.height - 2 * PAD)) * (DOMAIN.ymax - DOMAIN.ymin),
  ];

  function drawPlane() {
    const { width, height } = plane;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = palette.surface;
    ctx.fillRect(0, 0, width, height);
    // Decision regions of the FULL network (evaluation mode), tinted by
    // confidence.
    for (let sx = PAD; sx < width - PAD; sx += CELL)
      for (let sy = PAD; sy < height - PAD; sy += CELL) {
        const [wx, wy] = toWorld(sx + CELL / 2, sy + CELL / 2);
        const [z0, z1] = net.logits(wx, wy);
        const p1 = 1 / (1 + Math.exp(z0 - z1));
        ctx.globalAlpha = 0.08 + 0.22 * Math.abs(p1 - 0.5) * 2;
        ctx.fillStyle = p1 >= 0.5 ? palette.dataB : palette.dataA;
        ctx.fillRect(sx, sy, CELL, CELL);
      }
    ctx.globalAlpha = 1;
    // Training points: class by color AND shape (circle vs square); the
    // mislabeled ones get a ring so the reader can see them being memorized.
    for (const q of train) {
      const [sx, sy] = toScreen(q.x, q.y);
      ctx.fillStyle = q.label === 0 ? palette.dataA : palette.dataB;
      ctx.strokeStyle = palette.text;
      ctx.lineWidth = 1;
      ctx.beginPath();
      if (q.label === 0) ctx.arc(sx, sy, 4.5, 0, Math.PI * 2);
      else ctx.rect(sx - 4, sy - 4, 8, 8);
      ctx.fill();
      ctx.stroke();
      if (q.flipped) {
        ctx.beginPath();
        ctx.strokeStyle = palette.action;
        ctx.lineWidth = 1.5;
        ctx.arc(sx, sy, 8.5, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }

  function label(text: string, x: number, y: number, color = palette.muted, align: CanvasTextAlign = 'left') {
    sctx.fillStyle = color;
    sctx.font = '12px ui-sans-serif, system-ui, sans-serif';
    sctx.textAlign = align;
    sctx.fillText(text, x, y);
  }

  function drawSide() {
    const { width, height } = side;
    sctx.clearRect(0, 0, width, height);
    sctx.fillStyle = palette.surface;
    sctx.fillRect(0, 0, width, height);

    // Top half: this step's thinned network. Each hidden layer is an 8 x 8
    // grid of units -- filled if the unit survived, hollow if dropped.
    label(p === 0 ? 'every unit, every step (p = 0)' : `this step's thinned network (p = ${p})`, 12, 18);
    const cols = 8, cell = 15, gridW = cols * cell;
    const gx = [24, width - 24 - gridW];
    for (let l = 0; l < 2; l++) {
      const mask = net.lastMask[l];
      let kept = 0;
      for (let j = 0; j < HIDDEN; j++) {
        const cx = gx[l] + (j % cols) * cell + cell / 2;
        const cy = 36 + Math.floor(j / cols) * cell + cell / 2;
        sctx.beginPath();
        sctx.arc(cx, cy, 5, 0, Math.PI * 2);
        if (mask[j]) {
          kept++;
          sctx.fillStyle = palette.accent;
          sctx.fill();
        } else {
          sctx.strokeStyle = palette.axis;
          sctx.lineWidth = 1;
          sctx.stroke();
        }
      }
      label(`hidden ${l + 1}: ${kept}/${HIDDEN} on`, gx[l] + gridW / 2, 36 + 8 * cell + 14, palette.muted, 'center');
    }

    // Bottom half: train vs test accuracy over the run.
    const top = 190, bottom = height - 26, left = 34, right = width - 12;
    const yOf = (acc: number) => bottom - ((acc - 0.5) / 0.5) * (bottom - top);
    const xOf = (k: number) => left + (k / (MAX_STEPS / STEPS_PER_TICK)) * (right - left);
    sctx.strokeStyle = palette.axis;
    sctx.lineWidth = 1;
    sctx.beginPath();
    sctx.moveTo(left, top); sctx.lineTo(left, bottom); sctx.lineTo(right, bottom);
    sctx.stroke();
    for (const a of [0.5, 0.75, 1]) {
      label(`${Math.round(a * 100)}%`, left - 4, yOf(a) + 4, palette.muted, 'right');
      sctx.globalAlpha = 0.25;
      sctx.beginPath(); sctx.moveTo(left, yOf(a)); sctx.lineTo(right, yOf(a)); sctx.stroke();
      sctx.globalAlpha = 1;
    }
    const series: [keyof (typeof history)[number], string, number[]][] = [
      ['train', palette.text, []],
      ['test', palette.accent, [5, 4]],
    ];
    for (const [key, color, dash] of series) {
      sctx.strokeStyle = color;
      sctx.lineWidth = 2;
      sctx.setLineDash(dash);
      sctx.beginPath();
      history.forEach((h, k) => {
        const [x, y] = [xOf(k), yOf(Math.max(0.5, h[key]))];
        if (k === 0) sctx.moveTo(x, y); else sctx.lineTo(x, y);
      });
      sctx.stroke();
    }
    sctx.setLineDash([]);
    // Legend in the chart's empty bottom band (both curves live above ~65%).
    label('— train', right - 118, bottom - 24, palette.text);
    label('- - test (fresh points)', right - 118, bottom - 8, palette.accent);
    label(`step ${steps}`, left, height - 8);
  }

  function draw() {
    drawPlane();
    drawSide();
  }

  function emit() {
    const last = history[history.length - 1];
    opts.onStats?.({
      step: steps, trainAcc: last.train, testAcc: last.test, p, playing,
      done: steps >= MAX_STEPS,
    });
  }

  function tick() {
    for (let k = 0; k < STEPS_PER_TICK && steps < MAX_STEPS; k++) {
      net.step(train, p, dropRand);
      steps++;
    }
    history.push({ train: accuracy(net, train), test: accuracy(net, test) });
    if (steps >= MAX_STEPS) playing = false;
    draw();
    emit();
  }

  let last = 0;
  function loop(now: number) {
    if (!playing) return;
    if (now - last >= TICK_MS) {
      last = now;
      tick();
    }
    if (playing) raf = requestAnimationFrame(loop);
  }

  function play() {
    if (reducedMotion || steps >= MAX_STEPS) return;
    playing = true;
    last = 0;
    raf = requestAnimationFrame(loop);
    emit();
  }

  function pause() {
    playing = false;
    cancelAnimationFrame(raf);
    emit();
  }

  function onTheme() {
    palette = readPalette();
    draw();
  }
  window.addEventListener('themechange', onTheme);

  restart();
  draw();
  emit();

  return {
    toggle() { playing ? pause() : play(); },
    step() { pause(); if (steps < MAX_STEPS) tick(); },
    reset() { pause(); restart(); draw(); emit(); },
    setDrop(v: number) { pause(); p = v; restart(); draw(); emit(); },
    destroy() {
      pause();
      window.removeEventListener('themechange', onTheme);
    },
    reducedMotion,
  };
}
