// Module 06's interactive: how far back can a recurrent net remember?
//
// A vanilla tanh RNN learns the ECHO task live: a stream of random symbols from
// a 4-letter alphabet, and at every step the target is the symbol read d steps
// earlier. The reader picks the delay d and how far back the training gradient
// is allowed to reach (full backpropagation through time, or truncated every k
// steps). The lesson: the network learns a delay only if the gradient reaches
// back that far -- truncating at k steps makes every delay d >= k unlearnable
// -- and even full BPTT fades at long delays, because the gradient through
// time is a product of Jacobians diag(1 - h^2) Whh^T and shrinks geometrically.
//
//   h<t> = tanh(x<t> Wxh + h<t-1> Whh + bh)        h<-1> = 0
//   logits<t> = h<t> Why + bo                      loss = mean CE over t >= d
//   dh<t> = dH<t> + dz<t+1> Whh^T                  dz<t> = dh<t> * (1 - h<t>^2)
//   (no gradient from step t to t-1 when truncate > 0 and t % truncate == 0)
//
// The math is topics/06-rnn/python/rnn.py (build_echo_net, echo_loss_and_grads,
// echo_accuracy, grad_through_time) and lib/python/nanograd/rnn.py: same
// architecture, same Xavier draw order (Wxh, Whh, Why), same truncation rule,
// same Adam + gradient-norm clipping. Only the RNG differs (mulberry32 here),
// so the numbers differ from the NumPy run but the qualitative picture does not.
//
// The training half of this file is pure (no DOM): the verification script
// drives EchoRun directly. The rendering half mounts five canvases.
//
// Widget conventions (docs/conventions/viz-style.md): theme colors from CSS
// variables, repaint on `themechange`, symbols encoded by shape AND color,
// respect prefers-reduced-motion (no autoplay; Step advances 25 updates),
// deterministic seed + a reset control.

// ── Tunables ─────────────────────────────────────────────────────────────────
export const SEED = 3;           // weight draws
export const DATA_SEED = 11;     // training batches
export const TEST_SEED = 12;     // the fixed test set
export const PROFILE_SEED = 13;  // gradient-through-time inputs and directions
export const TAPE_SEED = 14;     // the one sequence drawn on the tape
export const V = 4;              // alphabet size (one-hot inputs)
export const NH = 16;            // hidden units
export const BATCH = 32;         // sequences per update
export const LR = 0.01;          // Adam
export const BETA1 = 0.9;
export const BETA2 = 0.999;
export const ADAM_EPS = 1e-8;
export const CLIP = 1.0;         // global gradient-norm clip (Pascanu et al. 2013)
export const MAX_STEPS = 3000;   // the run stops here (curves stay on screen)
export const EVAL_EVERY = 25;    // test accuracy / gradient profile refresh
export const TEST_T = 96;        // test set: 64 sequences of 96 steps
export const TEST_M = 64;
export const PROFILE_T = 40;     // gradient profile: 64 sequences of 40 steps
export const PROFILE_M = 64;
export const K_MAX = 32;         // gradient profile covers k = 0 .. 32 steps back
export const TAPE_T = 32;        // steps drawn on the tape / heatmap
export const DELAY_CHOICES = [1, 2, 4, 8, 12, 16] as const;
export const DEFAULT_DELAY = 4;
/** Gradient reach: 0 = full BPTT, k = cut the gradient every k steps. */
export const REACH_CHOICES = [1, 4, 8, 0] as const;
export const DEFAULT_REACH = 0;
const STEPS_PER_TICK = 4;        // at most this many updates per animation frame
const TICK_BUDGET_MS = 30;       // ...and stop early once a frame's updates took this long

/** Training sequence length for delay d: long enough to see many targets. */
export const trainLength = (d: number) => Math.max(24, 2 * d + 16);

// ── Randomness ───────────────────────────────────────────────────────────────

/** Tiny deterministic PRNG (mulberry32), as in the other widgets. */
export function mulberry32(seed: number): () => number {
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

/** T x m random symbols, time-major: symbol of sequence i at step t is S[t*m + i]. */
export function makeSeqs(T: number, m: number, rand: () => number): Int32Array {
  const S = new Int32Array(T * m);
  for (let j = 0; j < S.length; j++) S[j] = Math.floor(rand() * V);
  return S;
}

/** Xavier normal: std = sqrt(1 / fan_in), as nanograd's xavier_normal. */
function xavier(nIn: number, nOut: number, rand: () => number): Float64Array {
  const W = new Float64Array(nIn * nOut);
  const std = Math.sqrt(1 / nIn);
  for (let i = 0; i < W.length; i++) W[i] = std * gauss(rand);
  return W;
}

// ── The network: RNN(4 -> 16) + Linear(16 -> 4) ──────────────────────────────

/** Loss and gradients of one batch, in params() order. */
interface BatchResult { loss: number; grads: Float64Array[] }

export class EchoNet {
  /** Wxh: V x NH; Whh: NH x NH; Why: NH x V (row-major, rows = fan-in). */
  Wxh: Float64Array; Whh: Float64Array; bh = new Float64Array(NH);
  Why: Float64Array; bo = new Float64Array(V);

  /** Draw order: Wxh, Whh, Why from one stream; biases zero. */
  constructor(seed: number) {
    const rand = mulberry32(seed);
    this.Wxh = xavier(V, NH, rand);
    this.Whh = xavier(NH, NH, rand);
    this.Why = xavier(NH, V, rand);
  }

  /** Every parameter array, canonical order (the gradient list matches). */
  params(): Float64Array[] {
    return [this.Wxh, this.Whh, this.bh, this.Why, this.bo];
  }

  /** Forward through time. S: T x m symbols (time-major). Returns every hidden
   * state, H[(t*m + i)*NH + j] -- the recurrence is the only loop that cannot be
   * batched. h<-1> = 0. One-hot x Wxh is just a row lookup of Wxh. */
  forward(S: Int32Array, T: number, m: number): Float64Array {
    const H = new Float64Array(T * m * NH);
    const z = new Float64Array(NH);
    for (let t = 0; t < T; t++)
      for (let i = 0; i < m; i++) {
        const o = (t * m + i) * NH;
        const xo = S[t * m + i] * NH;
        for (let j = 0; j < NH; j++) z[j] = this.Wxh[xo + j] + this.bh[j];
        if (t > 0) {
          const po = ((t - 1) * m + i) * NH;
          for (let k = 0; k < NH; k++) {
            const h = H[po + k];
            const wo = k * NH;
            for (let j = 0; j < NH; j++) z[j] += h * this.Whh[wo + j];
          }
        }
        for (let j = 0; j < NH; j++) H[o + j] = Math.tanh(z[j]);
      }
    return H;
  }

  /** Read-out argmax at every (t, i): logits = h Why + bo. */
  predict(H: Float64Array, T: number, m: number): Uint8Array {
    const pred = new Uint8Array(T * m);
    for (let p = 0; p < T * m; p++) {
      let best = 0, bestZ = -Infinity;
      for (let c = 0; c < V; c++) {
        let s = this.bo[c];
        for (let k = 0; k < NH; k++) s += H[p * NH + k] * this.Why[k * V + c];
        if (s > bestZ) { bestZ = s; best = c; }
      }
      pred[p] = best;
    }
    return pred;
  }

  /** Echo loss and gradients on one batch, written out by hand: forward, then
   * softmax cross-entropy averaged over every position t >= d (no target before
   * that), then backpropagation through time with the truncation rule.
   * Does not touch the weights. */
  lossAndGrads(S: Int32Array, T: number, m: number, d: number, truncate: number): BatchResult {
    const H = this.forward(S, T, m);
    const cnt = (T - d) * m;

    // Read-out: dL/dlogits = (P - onehot)/cnt, which also gives dH at t >= d.
    const dH = new Float64Array(T * m * NH);
    const gWhy = new Float64Array(NH * V);
    const gbo = new Float64Array(V);
    const dlog = new Float64Array(V);
    let total = 0;
    for (let t = d; t < T; t++)
      for (let i = 0; i < m; i++) {
        const o = (t * m + i) * NH;
        let mx = -Infinity;
        for (let c = 0; c < V; c++) {
          let s = this.bo[c];
          for (let k = 0; k < NH; k++) s += H[o + k] * this.Why[k * V + c];
          dlog[c] = s;
          if (s > mx) mx = s;
        }
        let sum = 0;
        for (let c = 0; c < V; c++) { dlog[c] = Math.exp(dlog[c] - mx); sum += dlog[c]; }
        const y = S[(t - d) * m + i];
        total += -Math.log(dlog[y] / sum + 1e-12);
        for (let c = 0; c < V; c++) dlog[c] = (dlog[c] / sum - (c === y ? 1 : 0)) / cnt;
        for (let c = 0; c < V; c++) {
          gbo[c] += dlog[c];
          for (let k = 0; k < NH; k++) {
            gWhy[k * V + c] += H[o + k] * dlog[c];
            dH[o + k] += dlog[c] * this.Why[k * V + c];
          }
        }
      }

    // BPTT: dh = dH<t> + (gradient arriving from t+1); back through tanh;
    // the shared weights collect their share from every step (summed).
    const gWxh = new Float64Array(V * NH);
    const gWhh = new Float64Array(NH * NH);
    const gbh = new Float64Array(NH);
    const dhNext = new Float64Array(m * NH);
    const dz = new Float64Array(NH);
    for (let t = T - 1; t >= 0; t--)
      for (let i = 0; i < m; i++) {
        const o = (t * m + i) * NH;
        const no = i * NH;
        for (let j = 0; j < NH; j++) {
          const h = H[o + j];
          dz[j] = (dH[o + j] + dhNext[no + j]) * (1 - h * h);
        }
        const xo = S[t * m + i] * NH;
        for (let j = 0; j < NH; j++) { gWxh[xo + j] += dz[j]; gbh[j] += dz[j]; }
        if (t > 0) {
          const po = ((t - 1) * m + i) * NH;
          for (let k = 0; k < NH; k++) {
            const hp = H[po + k];
            for (let j = 0; j < NH; j++) gWhh[k * NH + j] += hp * dz[j];
          }
        }
        // THE POINT OF THE MODULE. Going one step further into the past
        // multiplies by Whh^T (and diag(1 - h^2) at the next step), the SAME
        // matrix every time. Truncation simply declines to take the step.
        if (truncate > 0 && t % truncate === 0) {
          for (let k = 0; k < NH; k++) dhNext[no + k] = 0;
        } else {
          for (let k = 0; k < NH; k++) {
            let s = 0;
            for (let j = 0; j < NH; j++) s += dz[j] * this.Whh[k * NH + j];
            dhNext[no + k] = s;
          }
        }
      }
    return { loss: total / cnt, grads: [gWxh, gWhh, gbh, gWhy, gbo] };
  }

  /** Accuracy over every position t >= d of a sequence set. */
  accuracy(S: Int32Array, T: number, m: number, d: number): number {
    const pred = this.predict(this.forward(S, T, m), T, m);
    let ok = 0;
    for (let t = d; t < T; t++)
      for (let i = 0; i < m; i++) if (pred[t * m + i] === S[(t - d) * m + i]) ok++;
    return ok / ((T - d) * m);
  }

  /** How much of the gradient at the LAST step reaches each earlier step.
   * Send a unit-norm random gradient into the final hidden state only, walk
   * back through time, and return ||dL/dh<T-1-k>|| averaged over the batch for
   * k = 0 .. K_MAX (grad_through_time in the Python). With truncation the walk
   * stops at the cut and the farther entries are NaN (absent). */
  gradProfile(S: Int32Array, T: number, m: number, truncate: number, g: Float64Array): Float64Array {
    const H = this.forward(S, T, m);
    const sum = new Float64Array(K_MAX + 1);
    const cnt = new Float64Array(K_MAX + 1);
    const dh = new Float64Array(NH);
    const nxt = new Float64Array(NH);
    for (let i = 0; i < m; i++) {
      let nrm = 0;
      for (let k = 0; k < NH; k++) nrm += g[i * NH + k] * g[i * NH + k];
      nrm = Math.sqrt(nrm);
      for (let k = 0; k < NH; k++) dh[k] = g[i * NH + k] / nrm;
      for (let k = 0; k <= K_MAX; k++) {
        const t = T - 1 - k;
        let n2 = 0;
        for (let j = 0; j < NH; j++) n2 += dh[j] * dh[j];
        sum[k] += Math.sqrt(n2);
        cnt[k]++;
        if (truncate > 0 && t % truncate === 0) break;     // the cut
        const o = (t * m + i) * NH;
        // dz = dh * (1 - h^2); the next dh = dz Whh^T.
        for (let j = 0; j < NH; j++) { const h = H[o + j]; dh[j] *= 1 - h * h; }
        for (let k2 = 0; k2 < NH; k2++) {
          let s = 0;
          for (let j = 0; j < NH; j++) s += dh[j] * this.Whh[k2 * NH + j];
          nxt[k2] = s;
        }
        dh.set(nxt);
      }
    }
    const out = new Float64Array(K_MAX + 1);
    for (let k = 0; k <= K_MAX; k++) out[k] = cnt[k] === m ? sum[k] / m : NaN;
    return out;
  }
}

/** Unit-direction source for the gradient profile: m x NH Gaussian draws. */
function profileDirections(m: number, rand: () => number): Float64Array {
  const g = new Float64Array(m * NH);
  for (let j = 0; j < g.length; j++) g[j] = gauss(rand);
  return g;
}

/** One training run: network + Adam + data stream + fixed evaluation sets.
 * `truncate` is the gradient reach (0 = full BPTT). */
export class EchoRun {
  readonly delay: number;
  readonly truncate: number;
  readonly net: EchoNet;
  steps = 0;
  /** Training loss of the most recent update. */
  loss = Math.log(V);
  /** Gradient norm before clipping, most recent update. */
  gradNorm = 0;

  private readonly T: number;
  private readonly dataRand = mulberry32(DATA_SEED);
  private readonly testS: Int32Array;
  private readonly profS: Int32Array;
  private readonly profG: Float64Array;
  private readonly mom: Float64Array[];
  private readonly vel: Float64Array[];

  constructor(delay: number, truncate: number, seed = SEED) {
    this.delay = delay;
    this.truncate = truncate;
    this.net = new EchoNet(seed);
    this.T = trainLength(delay);
    this.testS = makeSeqs(TEST_T, TEST_M, mulberry32(TEST_SEED));
    const pr = mulberry32(PROFILE_SEED);
    this.profS = makeSeqs(PROFILE_T, PROFILE_M, pr);
    this.profG = profileDirections(PROFILE_M, pr);
    this.mom = this.net.params().map((p) => new Float64Array(p.length));
    this.vel = this.net.params().map((p) => new Float64Array(p.length));
  }

  /** One update on a fresh random batch: gradients, global-norm clip, Adam. */
  update(): void {
    const S = makeSeqs(this.T, BATCH, this.dataRand);
    const { loss, grads } = this.net.lossAndGrads(S, this.T, BATCH, this.delay, this.truncate);
    this.loss = loss;

    // Clip: treat every gradient as one vector g; if ||g|| > 1, g *= 1/||g||.
    let total = 0;
    for (const g of grads) for (let j = 0; j < g.length; j++) total += g[j] * g[j];
    const norm = Math.sqrt(total);
    this.gradNorm = norm;
    if (norm > CLIP) {
      const scale = CLIP / norm;
      for (const g of grads) for (let j = 0; j < g.length; j++) g[j] *= scale;
    }

    // Adam with bias correction.
    this.steps++;
    const b1c = 1 - BETA1 ** this.steps;
    const b2c = 1 - BETA2 ** this.steps;
    const ps = this.net.params();
    for (let k = 0; k < ps.length; k++) {
      const P = ps[k], g = grads[k], m1 = this.mom[k], v2 = this.vel[k];
      for (let j = 0; j < P.length; j++) {
        m1[j] = BETA1 * m1[j] + (1 - BETA1) * g[j];
        v2[j] = BETA2 * v2[j] + (1 - BETA2) * g[j] * g[j];
        P[j] -= LR * (m1[j] / b1c) / (Math.sqrt(v2[j] / b2c) + ADAM_EPS);
      }
    }
  }

  /** Test accuracy on the fixed 64 x 96 set, every position t >= d. */
  evaluate(): number {
    return this.net.accuracy(this.testS, TEST_T, TEST_M, this.delay);
  }

  /** ||dL/dh<T-1-k>|| for k = 0 .. K_MAX on the fixed profile batch. */
  profile(): Float64Array {
    return this.net.gradProfile(this.profS, PROFILE_T, PROFILE_M, this.truncate, this.profG);
  }
}

// ── Rendering + interaction ──────────────────────────────────────────────────

interface Palette {
  surface: string; axis: string; text: string; muted: string;
  dataA: string; dataB: string; accent: string; action: string; pos: string; neg: string;
}
function readPalette(): Palette {
  const s = getComputedStyle(document.documentElement);
  const v = (n: string) => s.getPropertyValue(n).trim();
  return {
    surface: v('--color-surface'), axis: v('--viz-axis'), text: v('--color-text'),
    muted: v('--color-muted'), dataA: v('--viz-data-a'), dataB: v('--viz-data-b'),
    accent: v('--viz-accent'), action: v('--viz-action'),
    pos: v('--viz-pos'), neg: v('--viz-neg'),
  };
}

const LOSS_MIN = 1e-3, LOSS_MAX = 2;
const GRAD_MIN = 1e-6, GRAD_MAX = 1e2;
const LEFT = 64;                     // tape and heatmap share their column grid
const SHAPES = ['circle', 'square', 'triangle', 'diamond'] as const;

export interface EchoStats {
  step: number;
  delay: number;
  /** Gradient reach: 0 = full BPTT. */
  reach: number;
  acc: number;
  loss: number;
  playing: boolean;
  done: boolean;
}

export interface EchoHandles {
  toggle(): void;
  step(): void;
  reset(): void;
  setDelay(d: number): void;
  setReach(k: number): void;
  destroy(): void;
  reducedMotion: boolean;
}

export interface EchoCanvases {
  tape: HTMLCanvasElement;
  hidden: HTMLCanvasElement;
  acc: HTMLCanvasElement;
  loss: HTMLCanvasElement;
  grad: HTMLCanvasElement;
}

export function mountEcho(
  cv: EchoCanvases,
  opts: { onStats?: (s: EchoStats) => void; delay?: number; reach?: number } = {},
): EchoHandles {
  const ctx2 = (c: HTMLCanvasElement) => {
    const g = c.getContext('2d');
    if (!g) throw new Error('2D canvas context unavailable');
    return g;
  };
  const cTape = ctx2(cv.tape), cHid = ctx2(cv.hidden);
  const cAcc = ctx2(cv.acc), cLoss = ctx2(cv.loss), cGrad = ctx2(cv.grad);

  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let palette = readPalette();
  let delay: number = opts.delay ?? DEFAULT_DELAY;
  let reach: number = opts.reach ?? DEFAULT_REACH;
  let playing = false;
  let raf = 0;

  // One fixed sequence for the tape and heatmap (32 steps, so the first d
  // steps show "no target yet").
  const tapeS = makeSeqs(TAPE_T, 1, mulberry32(TAPE_SEED));

  let run: EchoRun;
  let curve: { step: number; acc: number }[] = [];
  let lossCurve: { step: number; loss: number }[] = [];
  let blockLoss = 0, blockN = 0;
  let evalStep = -1;
  let acc = 0;
  let prof: Float64Array = new Float64Array(K_MAX + 1);

  function restart() {
    run = new EchoRun(delay, reach);
    curve = [];
    lossCurve = [];
    blockLoss = 0;
    blockN = 0;
    evalStep = -1;
    ensureEval();
  }

  /** Refresh test accuracy and the gradient profile, and add a curve point. */
  function ensureEval() {
    if (evalStep === run.steps) return;
    evalStep = run.steps;
    acc = run.evaluate();
    prof = run.profile();
    curve.push({ step: run.steps, acc });
  }

  function text(g: CanvasRenderingContext2D, s: string, x: number, y: number,
                color = palette.muted, align: CanvasTextAlign = 'left') {
    g.fillStyle = color;
    g.font = '12px ui-sans-serif, system-ui, sans-serif';
    g.textAlign = align;
    g.fillText(s, x, y);
  }

  function clear(g: CanvasRenderingContext2D, c: HTMLCanvasElement) {
    g.clearRect(0, 0, c.width, c.height);
    g.fillStyle = palette.surface;
    g.fillRect(0, 0, c.width, c.height);
  }

  /** One symbol: a distinct shape AND a distinct color. */
  function glyph(g: CanvasRenderingContext2D, sym: number, cx: number, cy: number, r: number) {
    const colors = [palette.dataA, palette.dataB, palette.pos, palette.neg];
    g.fillStyle = colors[sym];
    g.strokeStyle = palette.text;
    g.lineWidth = 1;
    g.beginPath();
    switch (SHAPES[sym]) {
      case 'circle': g.arc(cx, cy, r, 0, Math.PI * 2); break;
      case 'square': g.rect(cx - r * 0.85, cy - r * 0.85, r * 1.7, r * 1.7); break;
      case 'triangle':
        g.moveTo(cx, cy - r * 1.1); g.lineTo(cx + r * 1.05, cy + r * 0.85);
        g.lineTo(cx - r * 1.05, cy + r * 0.85); g.closePath(); break;
      default:
        g.moveTo(cx, cy - r * 1.15); g.lineTo(cx + r * 1.15, cy);
        g.lineTo(cx, cy + r * 1.15); g.lineTo(cx - r * 1.15, cy); g.closePath();
    }
    g.fill();
    g.stroke();
  }

  const colW = (c: HTMLCanvasElement) => (c.width - 8 - LEFT) / TAPE_T;

  function drawTape(pred: Uint8Array) {
    const { width, height } = cv.tape;
    const g = cTape;
    clear(g, cv.tape);
    const w = colW(cv.tape);
    const cx = (t: number) => LEFT + (t + 0.5) * w;
    const rows = { inp: 44, tgt: 86, prd: 128 };
    const r = Math.min(4.2, w * 0.4);
    text(g, 'the tape — one test sequence', 8, 14, palette.text);
    text(g, 'input', LEFT - 8, rows.inp + 4, palette.muted, 'right');
    text(g, `target (d=${delay} back)`, LEFT - 8, rows.tgt + 4, palette.muted, 'right');
    text(g, 'prediction', LEFT - 8, rows.prd + 4, palette.muted, 'right');
    g.strokeStyle = palette.axis;
    g.lineWidth = 1;
    g.globalAlpha = 0.25;
    for (const y of [rows.inp, rows.tgt, rows.prd]) {
      g.beginPath(); g.moveTo(LEFT, y); g.lineTo(width - 8, y); g.stroke();
    }
    g.globalAlpha = 1;
    for (let t = 0; t < TAPE_T; t++) {
      glyph(g, tapeS[t], cx(t), rows.inp, r);
      if (t < delay) {           // no target yet: nothing to echo
        text(g, '·', cx(t), rows.tgt + 4, palette.muted, 'center');
        text(g, '·', cx(t), rows.prd + 4, palette.muted, 'center');
        continue;
      }
      glyph(g, tapeS[t - delay], cx(t), rows.tgt, r);
      glyph(g, pred[t], cx(t), rows.prd, r);
      if (pred[t] !== tapeS[t - delay]) {    // wrong: ring + cross, not color alone
        g.strokeStyle = palette.action;
        g.lineWidth = 2;
        g.beginPath();
        g.rect(cx(t) - w / 2 + 1, rows.prd - 9, w - 2, 18);
        g.stroke();
        text(g, '✗', cx(t), rows.prd + 24, palette.action, 'center');
      }
    }
    // The d-step offset: this input symbol becomes the target d steps later.
    const c0 = Math.max(0, TAPE_T - 2 - delay);
    const x0 = cx(c0), y0 = rows.inp + 9, x1 = cx(c0 + delay), y1 = rows.tgt - 9;
    g.strokeStyle = palette.text;
    g.fillStyle = palette.text;
    g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
    const ang = Math.atan2(y1 - y0, x1 - x0);
    g.beginPath();
    g.moveTo(x1, y1);
    g.lineTo(x1 - 7 * Math.cos(ang - 0.4), y1 - 7 * Math.sin(ang - 0.4));
    g.lineTo(x1 - 7 * Math.cos(ang + 0.4), y1 - 7 * Math.sin(ang + 0.4));
    g.closePath(); g.fill();
    const label = `d = ${delay}`;
    const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
    g.fillStyle = palette.surface;
    g.fillRect(mx - 22, my - 8, 44, 16);
    text(g, label, mx, my + 4, palette.text, 'center');
    text(g, 't = 0', LEFT, height - 8);
    text(g, `t = ${TAPE_T - 1}`, width - 8, height - 8, palette.muted, 'right');
  }

  function drawHidden(H: Float64Array) {
    const { width, height } = cv.hidden;
    const g = cHid;
    clear(g, cv.hidden);
    const w = colW(cv.hidden);
    const top = 22, ch = 7;
    text(g, 'hidden state h (16 units × 32 steps)', 8, 14, palette.text);
    for (let u = 0; u < NH; u++)
      for (let t = 0; t < TAPE_T; t++) {
        const v = H[t * NH + u];
        g.globalAlpha = Math.min(1, Math.abs(v));
        g.fillStyle = v >= 0 ? palette.pos : palette.neg;
        g.fillRect(LEFT + t * w, top + u * ch, Math.ceil(w), ch);
      }
    g.globalAlpha = 1;
    g.strokeStyle = palette.axis;
    g.lineWidth = 1;
    g.strokeRect(LEFT, top, TAPE_T * w, NH * ch);
    text(g, 'unit 1', LEFT - 6, top + 9, palette.muted, 'right');
    text(g, 'unit 16', LEFT - 6, top + NH * ch - 1, palette.muted, 'right');
    // Legend: sign is also readable from the +/- labels, not hue alone.
    const ly = top + NH * ch + 16;
    g.fillStyle = palette.pos; g.fillRect(LEFT, ly - 8, 10, 8);
    text(g, '+1', LEFT + 14, ly, palette.muted);
    g.fillStyle = palette.neg; g.fillRect(LEFT + 44, ly - 8, 10, 8);
    text(g, '−1', LEFT + 58, ly, palette.muted);
    text(g, '0 = blank', LEFT + 90, ly, palette.muted);
    text(g, `t = ${TAPE_T - 1}`, width - 8, height - 8, palette.muted, 'right');
  }

  const L10 = Math.log10;

  function frame(g: CanvasRenderingContext2D, c: HTMLCanvasElement, title: string,
                 bottomPad: number) {
    clear(g, c);
    const left = 44, right = c.width - 12, top = 24, bottom = c.height - bottomPad;
    text(g, title, left, 14, palette.text);
    g.strokeStyle = palette.axis;
    g.lineWidth = 1;
    g.beginPath(); g.moveTo(left, top); g.lineTo(left, bottom); g.lineTo(right, bottom); g.stroke();
    return { left, right, top, bottom };
  }

  function drawAcc() {
    const g = cAcc;
    const { left, right, top, bottom } = frame(g, cv.acc, 'test accuracy (every position t ≥ d)', 24);
    const yOf = (v: number) => bottom - v * (bottom - top);
    const xOf = (k: number) => left + (k / MAX_STEPS) * (right - left);
    for (const v of [0, 0.5, 1]) {
      text(g, `${Math.round(v * 100)}%`, left - 4, yOf(v) + 4, palette.muted, 'right');
      g.globalAlpha = 0.25;
      g.strokeStyle = palette.axis;
      g.beginPath(); g.moveTo(left, yOf(v)); g.lineTo(right, yOf(v)); g.stroke();
      g.globalAlpha = 1;
    }
    g.strokeStyle = palette.muted;      // chance: dashed, labelled
    g.lineWidth = 1;
    g.setLineDash([4, 4]);
    g.beginPath(); g.moveTo(left, yOf(0.25)); g.lineTo(right, yOf(0.25)); g.stroke();
    g.setLineDash([]);
    text(g, 'chance 25%', right, yOf(0.25) - 4, palette.muted, 'right');
    text(g, '0', left, cv.acc.height - 8);
    text(g, `${MAX_STEPS} updates`, right, cv.acc.height - 8, palette.muted, 'right');
    g.strokeStyle = palette.accent;
    g.lineWidth = 2;
    g.beginPath();
    curve.forEach((p, k) => { if (k === 0) g.moveTo(xOf(p.step), yOf(p.acc)); else g.lineTo(xOf(p.step), yOf(p.acc)); });
    g.stroke();
    const last = curve[curve.length - 1];
    g.fillStyle = palette.accent;
    g.beginPath(); g.arc(xOf(last.step), yOf(last.acc), 3.5, 0, Math.PI * 2); g.fill();
  }

  function drawLoss() {
    const g = cLoss;
    const { left, right, top, bottom } = frame(g, cv.loss, 'training loss (log scale, mean of 25 updates)', 24);
    const yOf = (v: number) => {
      const t = (L10(Math.min(LOSS_MAX, Math.max(LOSS_MIN, v))) - L10(LOSS_MIN)) / (L10(LOSS_MAX) - L10(LOSS_MIN));
      return bottom - t * (bottom - top);
    };
    const xOf = (k: number) => left + (k / MAX_STEPS) * (right - left);
    for (const v of [1e-3, 1e-2, 1e-1, 1]) {
      text(g, v >= 1 ? '1' : `1e${Math.round(L10(v))}`, left - 4, yOf(v) + 4, palette.muted, 'right');
      g.globalAlpha = 0.25;
      g.strokeStyle = palette.axis;
      g.beginPath(); g.moveTo(left, yOf(v)); g.lineTo(right, yOf(v)); g.stroke();
      g.globalAlpha = 1;
    }
    g.setLineDash([2, 3]);              // ln 4: loss of guessing uniformly
    g.strokeStyle = palette.muted;
    g.beginPath(); g.moveTo(left, yOf(Math.log(V))); g.lineTo(right, yOf(Math.log(V))); g.stroke();
    g.setLineDash([]);
    text(g, 'ln 4', right, yOf(Math.log(V)) - 4, palette.muted, 'right');
    text(g, '0', left, cv.loss.height - 8);
    text(g, `${MAX_STEPS} updates`, right, cv.loss.height - 8, palette.muted, 'right');
    g.strokeStyle = palette.text;
    g.lineWidth = 2;
    g.beginPath();
    lossCurve.forEach((p, k) => { if (k === 0) g.moveTo(xOf(p.step), yOf(p.loss)); else g.lineTo(xOf(p.step), yOf(p.loss)); });
    g.stroke();
  }

  function drawGrad() {
    const g = cGrad;
    const { left, right, top, bottom } = frame(g, cv.grad, '‖∂L/∂h‖ reaching k steps back from the last step', 30);
    const yOf = (v: number) => {
      const t = (L10(Math.min(GRAD_MAX, Math.max(GRAD_MIN, v))) - L10(GRAD_MIN)) / (L10(GRAD_MAX) - L10(GRAD_MIN));
      return bottom - t * (bottom - top);
    };
    for (let e = -6; e <= 2; e += 2) {
      text(g, e === 0 ? '1' : `1e${e}`, left - 4, yOf(10 ** e) + 4, palette.muted, 'right');
      g.globalAlpha = 0.25;
      g.strokeStyle = palette.axis;
      g.beginPath(); g.moveTo(left, yOf(10 ** e)); g.lineTo(right, yOf(10 ** e)); g.stroke();
      g.globalAlpha = 1;
    }
    const slot = (right - left) / (K_MAX + 1);
    const bw = Math.max(2, slot * 0.7);
    g.fillStyle = palette.dataA;
    for (let k = 0; k <= K_MAX; k++) {
      if (Number.isNaN(prof[k])) continue;           // beyond the cut: absent
      const y = yOf(prof[k]);
      g.fillRect(left + k * slot + (slot - bw) / 2, y, bw, bottom - y);
    }
    for (const k of [0, 8, 16, 24, 32])
      text(g, `${k}`, left + (k + 0.5) * slot, bottom + 14, palette.muted, 'center');
    text(g, 'k = steps back from the last step', right, cv.grad.height - 6, palette.muted, 'right');
    // The cut (dashed) and the delay d (solid), each labelled.
    let cutX = -1000;
    if (reach > 0 && reach <= K_MAX) {
      cutX = left + reach * slot;
      g.strokeStyle = palette.muted;
      g.lineWidth = 1.5;
      g.setLineDash([4, 3]);
      g.beginPath(); g.moveTo(cutX, top); g.lineTo(cutX, bottom); g.stroke();
      g.setLineDash([]);
      text(g, 'cut', cutX + 3, top + 11, palette.muted);
    }
    const dX = left + (delay + 0.5) * slot;
    g.strokeStyle = palette.action;
    g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(dX, top); g.lineTo(dX, bottom); g.stroke();
    text(g, 'd', dX + 3, top + (Math.abs(dX - cutX) < 24 ? 24 : 11), palette.action);
  }

  function draw() {
    const H = run.net.forward(tapeS, TAPE_T, 1);
    drawTape(run.net.predict(H, TAPE_T, 1));
    drawHidden(H);
    drawAcc();
    drawLoss();
    drawGrad();
  }

  function emit() {
    opts.onStats?.({
      step: run.steps, delay, reach, acc, loss: run.loss,
      playing, done: run.steps >= MAX_STEPS,
    });
  }

  function advance(n: number, budgetMs = Infinity, evalNow = true) {
    const t0 = performance.now();
    let crossed = false;
    for (let k = 0; k < n && run.steps < MAX_STEPS; k++) {
      if (k > 0 && performance.now() - t0 > budgetMs) break;
      run.update();
      blockLoss += run.loss;
      blockN++;
      if (run.steps % EVAL_EVERY === 0) {
        lossCurve.push({ step: run.steps, loss: blockLoss / blockN });
        blockLoss = 0; blockN = 0;
        crossed = true;
      }
    }
    if (run.steps >= MAX_STEPS) playing = false;
    if (crossed || evalNow || run.steps >= MAX_STEPS) ensureEval();
    draw();
    emit();
  }

  // Updates are cheap, so a frame takes up to STEPS_PER_TICK of them (within
  // the time budget): a full 3000-update run lasts about 12-15 s.
  function loop() {
    if (!playing) return;
    advance(STEPS_PER_TICK, TICK_BUDGET_MS, false);
    if (playing) raf = requestAnimationFrame(loop);
  }

  function play() {
    if (reducedMotion || run.steps >= MAX_STEPS) return;
    playing = true;
    raf = requestAnimationFrame(loop);
    emit();
  }

  function pause() {
    playing = false;
    cancelAnimationFrame(raf);
    ensureEval();
    draw();
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
    step() { pause(); advance(reducedMotion ? EVAL_EVERY : 1); },
    reset() { pause(); restart(); draw(); emit(); },
    setDelay(d: number) { pause(); delay = d; restart(); draw(); emit(); },
    setReach(k: number) { pause(); reach = k; restart(); draw(); emit(); },
    destroy() {
      pause();
      window.removeEventListener('themechange', onTheme);
    },
    reducedMotion,
  };
}
