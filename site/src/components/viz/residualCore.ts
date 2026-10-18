// Module 05's interactive: the degradation problem, and the shortcut that fixes it.
//
// Two networks of the SAME depth train side by side on two interleaved spirals,
// from the SAME initial weights. One is PLAIN; the other is its RESIDUAL twin.
// They differ in exactly one line: each residual block adds its input back,
// y = relu(F(x) + x). At a few blocks both fit all 128 points. Make them
// deeper and the plain net gets WORSE at fitting its own training data -- not
// overfitting, not a lack of capacity (extra blocks could compute the
// identity), but an optimization failure -- while the residual twin keeps
// fitting. The gradient chart shows why: in a deep plain net the gradient
// explodes toward the input (BatchNorm in a deep plain net does that); in the
// residual net it stays level, because the shortcut hands the gradient straight
// down, unchanged.
//
// The math is topics/05-resnet/python/resnet.py section 1 (make_spirals,
// ToyResNet): same data recipe, same architecture, same He draw order, same
// full-batch SGD + momentum. Only the RNG differs (mulberry32 here), so the
// numbers differ from the NumPy run but the qualitative picture does not.
//
// The training half of this file is pure (no DOM): the verification script
// drives ToyNet directly. The rendering half mounts four canvases.
//
// Widget conventions (docs/conventions/viz-style.md): theme colors from CSS
// variables, repaint on `themechange`, respect prefers-reduced-motion (no
// autoplay; Step advances 10 steps), deterministic seed + a reset control.

// ── Tunables ─────────────────────────────────────────────────────────────────
export const SEED = 2;           // weight draws (one stream, both twins)
export const DATA_SEED = 2;      // the spirals
export const N_TOY = 128;
export const TURNS = 1.5;
export const WIDTH = 16;
export const LR = 0.05;          // SGD
export const MOMENTUM = 0.9;
export const BN_EPS = 1e-5;
export const MAX_STEPS = 400;    // the run stops here (curves stay on screen)
export const DEPTH_CHOICES = [1, 4, 8, 16, 24] as const;
export const DEFAULT_DEPTH = 16;
const STEPS_PER_TICK = 4;        // at most this many SGD steps per animation frame
const TICK_BUDGET_MS = 30;       // ...and stop early once a frame's steps took this long
const CELL = 8;                  // decision-surface cell size, px

/** Weight layers for a net of `depth` blocks: stem + 2 per block + head. */
export const layerCount = (depth: number) => 2 * depth + 2;

// ── Data ─────────────────────────────────────────────────────────────────────

type Label = 0 | 1;
export interface Pt { x: number; y: number; label: Label }

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

/** Two interleaved spirals, class i % 2, one uniform draw t per point:
 * radius 0.15 + 0.85 t, angle 2 pi 1.5 t + pi class. No noise. */
export function makeSpirals(n: number, seed: number): Pt[] {
  const rand = mulberry32(seed);
  const pts: Pt[] = [];
  for (let i = 0; i < n; i++) {
    const label = (i % 2) as Label;
    const t = rand();
    const r = 0.15 + 0.85 * t;
    const a = 2 * Math.PI * TURNS * t + Math.PI * label;
    pts.push({ x: r * Math.cos(a), y: r * Math.sin(a), label });
  }
  return pts;
}

// ── Dense helpers (row-major, batch m) ───────────────────────────────────────

/** Y = X W for X: m x nIn, W: nIn x nOut (optional bias b). */
function matmul(X: Float64Array, W: Float64Array, m: number, nIn: number, nOut: number,
                b?: Float64Array): Float64Array {
  const Y = new Float64Array(m * nOut);
  for (let i = 0; i < m; i++) {
    const yo = i * nOut;
    if (b) for (let o = 0; o < nOut; o++) Y[yo + o] = b[o];
    // k outer, o inner: independent accumulators, W read along a row.
    for (let k = 0; k < nIn; k++) {
      const x = X[i * nIn + k];
      const wo = k * nOut;
      for (let o = 0; o < nOut; o++) Y[yo + o] += x * W[wo + o];
    }
  }
  return Y;
}

/** dW = X^T D for X: m x nIn, D: m x nOut. */
function matmulTA(X: Float64Array, D: Float64Array, m: number, nIn: number, nOut: number): Float64Array {
  const dW = new Float64Array(nIn * nOut);
  for (let i = 0; i < m; i++)
    for (let k = 0; k < nIn; k++) {
      const x = X[i * nIn + k];
      for (let o = 0; o < nOut; o++) dW[k * nOut + o] += x * D[i * nOut + o];
    }
  return dW;
}

/** dX = D W^T for D: m x nOut, W: nIn x nOut. */
function matmulTB(D: Float64Array, W: Float64Array, m: number, nIn: number, nOut: number): Float64Array {
  const dX = new Float64Array(m * nIn);
  for (let i = 0; i < m; i++)
    for (let k = 0; k < nIn; k++) {
      let s = 0;
      for (let o = 0; o < nOut; o++) s += D[i * nOut + o] * W[k * nOut + o];
      dX[i * nIn + k] = s;
    }
  return dX;
}

function relu(Z: Float64Array): Float64Array {
  const A = new Float64Array(Z.length);
  for (let j = 0; j < Z.length; j++) A[j] = Z[j] > 0 ? Z[j] : 0;
  return A;
}

/** Batch statistics a BatchNorm layer used on the most recent training step. */
interface BnStat { mean: Float64Array; inv: Float64Array }

/** BatchNorm forward over the batch, per unit (columns of Z: m x WIDTH).
 * Training: mean, biased variance, xhat = (z - mean)/sqrt(var + eps); the
 * statistics are stored in `stat`. Evaluation (decision surface): reuse the
 * stored statistics of the latest training batch. */
function bnForward(Z: Float64Array, m: number, gamma: Float64Array, beta: Float64Array,
                   stat: BnStat, training: boolean): { Y: Float64Array; XH: Float64Array } {
  const C = WIDTH;
  if (training) {
    for (let c = 0; c < C; c++) {
      let mean = 0;
      for (let i = 0; i < m; i++) mean += Z[i * C + c];
      mean /= m;
      let v = 0;
      for (let i = 0; i < m; i++) { const d = Z[i * C + c] - mean; v += d * d; }
      v /= m;
      stat.mean[c] = mean;
      stat.inv[c] = 1 / Math.sqrt(v + BN_EPS);
    }
  }
  const XH = new Float64Array(m * C);
  const Y = new Float64Array(m * C);
  for (let i = 0; i < m; i++)
    for (let c = 0; c < C; c++) {
      const xh = (Z[i * C + c] - stat.mean[c]) * stat.inv[c];
      XH[i * C + c] = xh;
      Y[i * C + c] = gamma[c] * xh + beta[c];
    }
  return { Y, XH };
}

/** BatchNorm backward (training mode): returns dZ and fills dGamma / dBeta.
 * dbeta = sum dy, dgamma = sum dy xhat,
 * dz = (gamma invstd / m)(m dy - dbeta - xhat dgamma). */
function bnBackward(XH: Float64Array, dY: Float64Array, m: number, gamma: Float64Array,
                    inv: Float64Array, dGamma: Float64Array, dBeta: Float64Array): Float64Array {
  const C = WIDTH;
  const dZ = new Float64Array(m * C);
  for (let c = 0; c < C; c++) {
    let sb = 0, sg = 0;
    for (let i = 0; i < m; i++) { sb += dY[i * C + c]; sg += dY[i * C + c] * XH[i * C + c]; }
    dBeta[c] = sb;
    dGamma[c] = sg;
    const kc = (gamma[c] * inv[c]) / m;
    for (let i = 0; i < m; i++)
      dZ[i * C + c] = kc * (m * dY[i * C + c] - sb - XH[i * C + c] * sg);
  }
  return dZ;
}

// ── The network: stem -> D blocks -> head, plain or residual ─────────────────

interface Block {
  W1: Float64Array; g1: Float64Array; be1: Float64Array;
  W2: Float64Array; g2: Float64Array; be2: Float64Array;
  s1: BnStat; s2: BnStat;
}

function heNormal(nIn: number, nOut: number, rand: () => number): Float64Array {
  const W = new Float64Array(nIn * nOut);
  const std = Math.sqrt(2 / nIn); // He init, as nanograd's he_normal
  for (let i = 0; i < W.length; i++) W[i] = std * gauss(rand);
  return W;
}

const newStat = (): BnStat => ({ mean: new Float64Array(WIDTH), inv: new Float64Array(WIDTH).fill(1) });

export class ToyNet {
  readonly residual: boolean;
  readonly depth: number;
  W0: Float64Array; b0 = new Float64Array(WIDTH);
  blocks: Block[] = [];
  Wh: Float64Array; bh = new Float64Array(2);
  /** Momentum velocities, one per parameter array, in params() order. */
  vel: Float64Array[];

  /** Results of the most recent forward/backward (computeGrads). */
  loss = Math.log(2);
  acc = 0.5;
  /** ||dW1|| of each block's first Linear, input side first. */
  gradNorms: Float64Array;

  /** Draw order: stem W0, then each block's W1 and W2, then head Wh -- one
   * He stream. Two nets built from the same seed start IDENTICAL; `residual`
   * switches the `+ x` on and changes nothing else. */
  constructor(seed: number, depth: number, residual: boolean) {
    this.residual = residual;
    this.depth = depth;
    const rand = mulberry32(seed);
    this.W0 = heNormal(2, WIDTH, rand);
    for (let d = 0; d < depth; d++) {
      const W1 = heNormal(WIDTH, WIDTH, rand);
      const W2 = heNormal(WIDTH, WIDTH, rand);
      this.blocks.push({
        W1, g1: new Float64Array(WIDTH).fill(1), be1: new Float64Array(WIDTH),
        W2, g2: new Float64Array(WIDTH).fill(1), be2: new Float64Array(WIDTH),
        s1: newStat(), s2: newStat(),
      });
    }
    this.Wh = heNormal(WIDTH, 2, rand);
    this.vel = this.params().map((p) => new Float64Array(p.length));
    this.gradNorms = new Float64Array(depth);
  }

  /** Every parameter array, canonical order (the gradient list matches). */
  params(): Float64Array[] {
    const ps = [this.W0, this.b0];
    for (const b of this.blocks) ps.push(b.W1, b.g1, b.be1, b.W2, b.g2, b.be2);
    ps.push(this.Wh, this.bh);
    return ps;
  }

  /** Forward pass over m points (X: m x 2). `training` computes batch
   * statistics (and stores them); otherwise BatchNorm reuses the stored ones.
   * Returns logits plus the caches backward needs. */
  private forward(X: Float64Array, m: number, training: boolean) {
    const H0 = matmul(X, this.W0, m, 2, WIDTH, this.b0);
    let A = relu(H0);
    const caches: { Ain: Float64Array; U1: Float64Array; XH1: Float64Array; A1: Float64Array;
                    XH2: Float64Array; S: Float64Array }[] = [];
    for (const b of this.blocks) {
      const Z1 = matmul(A, b.W1, m, WIDTH, WIDTH);
      const bn1 = bnForward(Z1, m, b.g1, b.be1, b.s1, training);
      const A1 = relu(bn1.Y);
      const Z2 = matmul(A1, b.W2, m, WIDTH, WIDTH);
      const bn2 = bnForward(Z2, m, b.g2, b.be2, b.s2, training);
      // The one difference between the twins: the shortcut adds the block's
      // input back before the final ReLU.
      let S = bn2.Y;
      if (this.residual) {
        S = new Float64Array(bn2.Y.length);
        for (let j = 0; j < S.length; j++) S[j] = bn2.Y[j] + A[j];
      }
      if (training) caches.push({ Ain: A, U1: bn1.Y, XH1: bn1.XH, A1, XH2: bn2.XH, S });
      A = relu(S);
    }
    return { logits: matmul(A, this.Wh, m, WIDTH, 2, this.bh), A, H0, caches };
  }

  /** Evaluation-mode logits for a batch (the decision surface): BatchNorm uses
   * the statistics of the most recent training batch, which keeps the picture
   * consistent with the training accuracy shown. */
  logitsFor(X: Float64Array, m: number): Float64Array {
    return this.forward(X, m, false).logits;
  }

  /** Full-batch forward + backward, written out by hand. Sets loss, acc,
   * gradNorms and the stored BatchNorm statistics; returns gradients in
   * params() order. Does not touch the weights. */
  computeGrads(X: Float64Array, labels: Uint8Array): Float64Array[] {
    const m = labels.length;
    const { logits, A, H0, caches } = this.forward(X, m, true);

    // Softmax cross-entropy (mean over the batch): dL/dlogits = (P - onehot)/m.
    let total = 0, ok = 0;
    const dlog = new Float64Array(m * 2);
    for (let i = 0; i < m; i++) {
      const z0 = logits[2 * i], z1 = logits[2 * i + 1];
      const mx = Math.max(z0, z1);
      const e0 = Math.exp(z0 - mx), e1 = Math.exp(z1 - mx);
      const s = e0 + e1;
      const p0 = e0 / s, p1 = e1 / s;
      const y = labels[i];
      total += -Math.log((y === 0 ? p0 : p1) + 1e-12);
      if ((z1 > z0 ? 1 : 0) === y) ok++;
      dlog[2 * i] = (p0 - (y === 0 ? 1 : 0)) / m;
      dlog[2 * i + 1] = (p1 - (y === 1 ? 1 : 0)) / m;
    }
    this.loss = total / m;
    this.acc = ok / m;

    // Head, then the blocks in reverse, then the stem.
    const gWh = matmulTA(A, dlog, m, WIDTH, 2);
    const gbh = new Float64Array(2);
    for (let i = 0; i < m; i++) { gbh[0] += dlog[2 * i]; gbh[1] += dlog[2 * i + 1]; }
    let dA = matmulTB(dlog, this.Wh, m, WIDTH, 2);

    const blockGrads: Float64Array[][] = [];
    for (let d = this.depth - 1; d >= 0; d--) {
      const b = this.blocks[d];
      const c = caches[d];
      // Through the final ReLU: dS = dA where S > 0.
      const dS = new Float64Array(m * WIDTH);
      for (let j = 0; j < dS.length; j++) dS[j] = c.S[j] > 0 ? dA[j] : 0;
      const gg2 = new Float64Array(WIDTH), gbe2 = new Float64Array(WIDTH);
      const dZ2 = bnBackward(c.XH2, dS, m, b.g2, b.s2.inv, gg2, gbe2);
      const gW2 = matmulTA(c.A1, dZ2, m, WIDTH, WIDTH);
      const dA1 = matmulTB(dZ2, b.W2, m, WIDTH, WIDTH);
      const dU1 = new Float64Array(m * WIDTH);
      for (let j = 0; j < dU1.length; j++) dU1[j] = c.U1[j] > 0 ? dA1[j] : 0;
      const gg1 = new Float64Array(WIDTH), gbe1 = new Float64Array(WIDTH);
      const dZ1 = bnBackward(c.XH1, dU1, m, b.g1, b.s1.inv, gg1, gbe1);
      const gW1 = matmulTA(c.Ain, dZ1, m, WIDTH, WIDTH);
      const dAin = matmulTB(dZ1, b.W1, m, WIDTH, WIDTH);

      let nrm = 0;
      for (let j = 0; j < gW1.length; j++) nrm += gW1[j] * gW1[j];
      this.gradNorms[d] = Math.sqrt(nrm);

      // THE POINT OF THE MODULE. The block's input fed two routes -- the body
      // (Linear, BN, ReLU, Linear, BN) and, in the residual twin, the shortcut
      // straight to the sum. Its gradient is the SUM of the gradient through
      // the body and the gradient straight down the shortcut. The shortcut
      // term dS arrives untouched by any weight matrix or BatchNorm, so no
      // matter how deep the stack, a path of "just add" carries the signal
      // back to the input. The plain net has only the body term, and every
      // block it passes through multiplies the gradient again.
      if (this.residual) {
        dA = new Float64Array(dAin.length);
        for (let j = 0; j < dA.length; j++) dA[j] = dAin[j] + dS[j];
      } else {
        dA = dAin;
      }
      blockGrads.unshift([gW1, gg1, gbe1, gW2, gg2, gbe2]);
    }
    const dH0 = new Float64Array(m * WIDTH);
    for (let j = 0; j < dH0.length; j++) dH0[j] = H0[j] > 0 ? dA[j] : 0;
    const gW0 = matmulTA(X, dH0, m, 2, WIDTH);
    const gb0 = new Float64Array(WIDTH);
    for (let i = 0; i < m; i++) for (let c = 0; c < WIDTH; c++) gb0[c] += dH0[i * WIDTH + c];

    return [gW0, gb0, ...blockGrads.flat(), gWh, gbh];
  }

  /** One full-batch SGD + momentum step: v = 0.9 v + g; theta -= lr v. */
  step(X: Float64Array, labels: Uint8Array): void {
    const grads = this.computeGrads(X, labels);
    const ps = this.params();
    for (let k = 0; k < ps.length; k++) {
      const P = ps[k], g = grads[k], v = this.vel[k];
      for (let j = 0; j < P.length; j++) {
        v[j] = MOMENTUM * v[j] + g[j];
        P[j] -= LR * v[j];
      }
    }
  }
}

/** The pair under test, from IDENTICAL initial weights (same seed, same draw
 * order); `residual` is the only difference. */
export function buildTwins(depth: number, seed = SEED): { plain: ToyNet; resid: ToyNet } {
  return { plain: new ToyNet(seed, depth, false), resid: new ToyNet(seed, depth, true) };
}

/** Pack points into the flat arrays the nets take. */
export function packPoints(pts: Pt[]): { X: Float64Array; labels: Uint8Array } {
  const X = new Float64Array(pts.length * 2);
  const labels = new Uint8Array(pts.length);
  pts.forEach((q, i) => { X[2 * i] = q.x; X[2 * i + 1] = q.y; labels[i] = q.label; });
  return { X, labels };
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

const DOMAIN = 1.15;   // both axes run -DOMAIN .. DOMAIN
const PAD = 6;
const LOSS_MIN = 1e-4, LOSS_MAX = 2;
const GRAD_MIN = 1e-6, GRAD_MAX = 1e5;

export interface ResidualStats {
  step: number;
  depth: number;
  layers: number;
  plainAcc: number;
  residAcc: number;
  plainLoss: number;
  residLoss: number;
  playing: boolean;
  done: boolean;
}

export interface ResidualHandles {
  toggle(): void;
  step(): void;
  reset(): void;
  setDepth(d: number): void;
  destroy(): void;
  reducedMotion: boolean;
}

export interface ResidualCanvases {
  plain: HTMLCanvasElement;
  resid: HTMLCanvasElement;
  loss: HTMLCanvasElement;
  grad: HTMLCanvasElement;
}

export function mountResidual(
  cv: ResidualCanvases,
  opts: { onStats?: (s: ResidualStats) => void; depth?: number } = {},
): ResidualHandles {
  const ctx2 = (c: HTMLCanvasElement) => {
    const g = c.getContext('2d');
    if (!g) throw new Error('2D canvas context unavailable');
    return g;
  };
  const cPlain = ctx2(cv.plain), cResid = ctx2(cv.resid);
  const cLoss = ctx2(cv.loss), cGrad = ctx2(cv.grad);

  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let palette = readPalette();
  let depth: number = opts.depth ?? DEFAULT_DEPTH;
  let playing = false;
  let raf = 0;

  // The data is fixed by the seed; the nets restart from the same init.
  const train = makeSpirals(N_TOY, DATA_SEED);
  const { X, labels } = packPoints(train);

  let plain: ToyNet, resid: ToyNet;
  let steps = 0;
  let lossP: number[] = [];
  let lossR: number[] = [];
  // Each net's gradient profile at step 0, kept on screen as tick marks: the
  // plain net's explosion is an initialization effect, and training levels it
  // (or the fit drives every gradient toward zero) long before step 400.
  let initP = new Float64Array(0);
  let initR = new Float64Array(0);

  // The decision-surface grid: one batched forward pass per net per redraw.
  const side = cv.plain.width;
  const cols = Math.floor((side - 2 * PAD) / CELL);
  const gridX = new Float64Array(cols * cols * 2);
  for (let r = 0; r < cols; r++)
    for (let c = 0; c < cols; c++) {
      const k = r * cols + c;
      gridX[2 * k] = -DOMAIN + ((c + 0.5) / cols) * 2 * DOMAIN;
      gridX[2 * k + 1] = DOMAIN - ((r + 0.5) / cols) * 2 * DOMAIN;
    }

  function restart() {
    ({ plain, resid } = buildTwins(depth));
    // Step 0: forward + backward without an update, so the surfaces, the
    // BatchNorm statistics and the gradient bars exist before the first step.
    plain.computeGrads(X, labels);
    resid.computeGrads(X, labels);
    initP = plain.gradNorms.slice();
    initR = resid.gradNorms.slice();
    steps = 0;
    lossP = [plain.loss];
    lossR = [resid.loss];
  }

  function drawSurface(g: CanvasRenderingContext2D, canvas: HTMLCanvasElement, net: ToyNet, title: string) {
    const { width, height } = canvas;
    g.clearRect(0, 0, width, height);
    g.fillStyle = palette.surface;
    g.fillRect(0, 0, width, height);
    const inner = width - 2 * PAD;
    const cell = inner / cols;
    const Z = net.logitsFor(gridX, cols * cols);
    for (let r = 0; r < cols; r++)
      for (let c = 0; c < cols; c++) {
        const k = r * cols + c;
        const p1 = 1 / (1 + Math.exp(Z[2 * k] - Z[2 * k + 1]));
        g.globalAlpha = 0.08 + 0.3 * Math.abs(p1 - 0.5) * 2;
        g.fillStyle = p1 >= 0.5 ? palette.dataB : palette.dataA;
        g.fillRect(PAD + c * cell, PAD + r * cell, Math.ceil(cell), Math.ceil(cell));
      }
    g.globalAlpha = 1;
    // Training points: class by color AND shape (circle vs square).
    for (const q of train) {
      const sx = PAD + ((q.x + DOMAIN) / (2 * DOMAIN)) * inner;
      const sy = height - PAD - ((q.y + DOMAIN) / (2 * DOMAIN)) * inner;
      g.fillStyle = q.label === 0 ? palette.dataA : palette.dataB;
      g.strokeStyle = palette.text;
      g.lineWidth = 1;
      g.beginPath();
      if (q.label === 0) g.arc(sx, sy, 3.6, 0, Math.PI * 2);
      else g.rect(sx - 3.2, sy - 3.2, 6.4, 6.4);
      g.fill();
      g.stroke();
    }
    g.fillStyle = palette.text;
    g.font = '12px ui-sans-serif, system-ui, sans-serif';
    g.textAlign = 'left';
    g.fillText(title, PAD + 4, PAD + 12);
    g.fillStyle = palette.muted;
    g.fillText(`${Math.round(net.acc * 100)}% fit`, PAD + 4, height - PAD - 4);
  }

  function text(g: CanvasRenderingContext2D, s: string, x: number, y: number,
                color = palette.muted, align: CanvasTextAlign = 'left') {
    g.fillStyle = color;
    g.font = '12px ui-sans-serif, system-ui, sans-serif';
    g.textAlign = align;
    g.fillText(s, x, y);
  }

  const L10 = Math.log10;

  function drawLoss() {
    const { width, height } = cv.loss;
    const g = cLoss;
    g.clearRect(0, 0, width, height);
    g.fillStyle = palette.surface;
    g.fillRect(0, 0, width, height);
    const left = 44, right = width - 12, top = 24, bottom = height - 24;
    const yOf = (v: number) => {
      const t = (L10(Math.min(LOSS_MAX, Math.max(LOSS_MIN, v))) - L10(LOSS_MIN)) / (L10(LOSS_MAX) - L10(LOSS_MIN));
      return bottom - t * (bottom - top);
    };
    const xOf = (k: number) => left + (k / MAX_STEPS) * (right - left);
    text(g, 'training loss (log scale)', left, 14, palette.text);
    g.strokeStyle = palette.axis;
    g.lineWidth = 1;
    g.beginPath(); g.moveTo(left, top); g.lineTo(left, bottom); g.lineTo(right, bottom); g.stroke();
    for (const v of [1e-4, 1e-3, 1e-2, 1e-1, 1]) {
      text(g, v >= 1 ? '1' : `1e${Math.round(L10(v))}`, left - 4, yOf(v) + 4, palette.muted, 'right');
      g.globalAlpha = 0.25;
      g.beginPath(); g.moveTo(left, yOf(v)); g.lineTo(right, yOf(v)); g.stroke();
      g.globalAlpha = 1;
    }
    text(g, '0', left, height - 8);
    text(g, `${MAX_STEPS} steps`, right, height - 8, palette.muted, 'right');
    const series: [number[], string, number[]][] = [[lossP, palette.text, [5, 4]], [lossR, palette.accent, []]];
    for (const [data, color, dash] of series) {
      g.strokeStyle = color;
      g.lineWidth = 2;
      g.setLineDash(dash);
      g.beginPath();
      data.forEach((v, k) => { if (k === 0) g.moveTo(xOf(k), yOf(v)); else g.lineTo(xOf(k), yOf(v)); });
      g.stroke();
    }
    g.setLineDash([]);
    // Legend, top right, clear of the curves' usual region.
    const lx = right - 150;
    g.strokeStyle = palette.text; g.lineWidth = 2; g.setLineDash([5, 4]);
    g.beginPath(); g.moveTo(lx, 12); g.lineTo(lx + 22, 12); g.stroke();
    g.setLineDash([]);
    text(g, 'plain', lx + 28, 16, palette.text);
    g.strokeStyle = palette.accent;
    g.beginPath(); g.moveTo(lx + 76, 12); g.lineTo(lx + 98, 12); g.stroke();
    text(g, 'residual', lx + 104, 16, palette.accent);
  }

  function drawGrad() {
    const { width, height } = cv.grad;
    const g = cGrad;
    g.clearRect(0, 0, width, height);
    g.fillStyle = palette.surface;
    g.fillRect(0, 0, width, height);
    const left = 44, right = width - 12, top = 24, bottom = height - 30;
    const yOf = (v: number) => {
      const t = (L10(Math.min(GRAD_MAX, Math.max(GRAD_MIN, v))) - L10(GRAD_MIN)) / (L10(GRAD_MAX) - L10(GRAD_MIN));
      return bottom - t * (bottom - top);
    };
    text(g, steps === 0 ? '‖dW₁‖ per block, step 0' : `‖dW₁‖ per block, step ${steps} (ticks: step 0)`,
         left, 14, palette.text);
    g.strokeStyle = palette.axis;
    g.lineWidth = 1;
    g.beginPath(); g.moveTo(left, top); g.lineTo(left, bottom); g.lineTo(right, bottom); g.stroke();
    for (let e = -5; e <= 5; e += 2) {
      text(g, `1e${e}`, left - 4, yOf(10 ** e) + 4, palette.muted, 'right');
      g.globalAlpha = 0.25;
      g.beginPath(); g.moveTo(left, yOf(10 ** e)); g.lineTo(right, yOf(10 ** e)); g.stroke();
      g.globalAlpha = 1;
    }
    // Two interleaved series per block: plain (hollow-ish text color) and
    // residual (accent). Input side on the left.
    const D = depth;
    const slot = (right - left) / D;
    const bw = Math.max(1.5, Math.min(10, slot * 0.42));
    for (let d = 0; d < D; d++) {
      const x0 = left + d * slot + slot / 2;
      const pairs: [number, number, number, string][] = [
        [plain.gradNorms[d], initP[d], x0 - bw - 0.5, palette.text],
        [resid.gradNorms[d], initR[d], x0 + 0.5, palette.accent],
      ];
      for (const [v, v0, x, color] of pairs) {
        const y = yOf(v);
        g.fillStyle = color;
        g.globalAlpha = color === palette.text ? 0.55 : 1;
        g.fillRect(x, y, bw, bottom - y);
        if (steps > 0) {                       // where this bar stood at step 0
          g.globalAlpha = 1;
          g.fillRect(x - 1, yOf(v0) - 1, bw + 2, 2);
        }
      }
      g.globalAlpha = 1;
    }
    text(g, 'block 1 (input side)', left, height - 8);
    text(g, `block ${D}`, right, height - 8, palette.muted, 'right');
    const lx = right - 118;
    g.fillStyle = palette.text; g.globalAlpha = 0.55; g.fillRect(lx, 7, 10, 8); g.globalAlpha = 1;
    text(g, 'plain', lx + 14, 16, palette.text);
    g.fillStyle = palette.accent; g.fillRect(lx + 52, 7, 10, 8);
    text(g, 'residual', lx + 66, 16, palette.accent);
  }

  function draw() {
    drawSurface(cPlain, cv.plain, plain, 'plain');
    drawSurface(cResid, cv.resid, resid, 'residual');
    drawLoss();
    drawGrad();
  }

  function emit() {
    opts.onStats?.({
      step: steps, depth, layers: layerCount(depth),
      plainAcc: plain.acc, residAcc: resid.acc,
      plainLoss: plain.loss, residLoss: resid.loss,
      playing, done: steps >= MAX_STEPS,
    });
  }

  function advance(n: number, budgetMs = Infinity) {
    const t0 = performance.now();
    for (let k = 0; k < n && steps < MAX_STEPS; k++) {
      if (k > 0 && performance.now() - t0 > budgetMs) break;
      plain.step(X, labels);
      resid.step(X, labels);
      steps++;
      lossP.push(plain.loss);
      lossR.push(resid.loss);
    }
    if (steps >= MAX_STEPS) playing = false;
    draw();
    emit();
  }

  // Deep nets are slower per step, so a frame takes as many steps as fit in
  // the budget (1..STEPS_PER_TICK): shallow nets race, depth 24 stays smooth.
  function loop() {
    if (!playing) return;
    advance(STEPS_PER_TICK, TICK_BUDGET_MS);
    if (playing) raf = requestAnimationFrame(loop);
  }

  function play() {
    if (reducedMotion || steps >= MAX_STEPS) return;
    playing = true;
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
    step() { pause(); advance(reducedMotion ? 10 : 1); },
    reset() { pause(); restart(); draw(); emit(); },
    setDepth(d: number) { pause(); depth = d; restart(); draw(); emit(); },
    destroy() {
      pause();
      window.removeEventListener('themechange', onTheme);
    },
    reducedMotion,
  };
}
