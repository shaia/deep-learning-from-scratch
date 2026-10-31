// Module 07's interactive: can a network hold one symbol across a long stretch of noise?
//
// Four recurrent networks learn the LATCH task live. Step 0 shows a SIGNAL (one
// of A B C D); steps 1..T-1 show DISTRACTORS (w x y z, uniformly random); at the
// last step the network must say which signal came first. The only error signal
// sits at the end, so the lesson "remember that symbol" must travel T-1 steps
// back in time to the step that stored it. The reader picks the network and the
// lag T:
//
//   RNN                      Module 06's tanh RNN, full BPTT
//   LSTM, forget bias 1      the textbook init: input gate bias 0, forget bias 1
//   LSTM, gates open         input gate bias -5 (shut to new input, i = 0.007),
//                            forget bias +5 (open to the past, f = 0.993)
//   LSTM, carousel only      the same biases, trained with Hochreiter &
//                            Schmidhuber's 1997 gradient: it travels back along
//                            the cell only, never through the gates' h<t-1>
//
// The LSTM (the four gate blocks side by side, in the order [i | f | g | o]):
//
//   z<t> = x<t> Wx + h<t-1> Wh + b          i, f, o = sigmoid(.)    g = tanh(.)
//   c<t> = f * c<t-1> + i * g               h<t> = o * tanh(c<t>)
//
//   dh<t> = dH<t> + dz<t+1> Wh^T            (through the gates, as the RNN)
//   dc<t> = dc<t+1> * f<t+1> + dh<t> * o * (1 - tanh(c<t>)^2)     (the carousel)
//   dz_i = dc g i(1-i)   dz_f = dc c<t-1> f(1-f)   dz_g = dc i (1-g^2)   dz_o = dh tanh(c) o(1-o)
//   (cell-only: dh_next = 0 at every step; the carousel dc_next = dc * f still flows)
//
// The lesson: along the cell one step back multiplies the gradient by f -- a
// number the network sets for itself -- so with f near 1 the gradient arrives
// from 160 steps back almost intact; the RNN's arrives as a product of Jacobians
// and is gone; the textbook forget bias of 1 keeps only ~3.7 steps of memory.
//
// The math is topics/07-lstm/python/lstm.py (latch_batch, build_latch_net,
// latch_loss_and_grads, train_latch, grad_through_time, gate_trace) and
// lib/python/nanograd/{lstm,rnn}.py: same architecture (16 units -> Linear(16 -> 4)
// on the last state), same Xavier draw order (input weights, recurrent weights,
// read-out), same biases, same Adam + gradient-norm clipping. Only the RNG
// differs (mulberry32 here), so the numbers differ from the NumPy run but the
// qualitative picture does not.
//
// The training half of this file is pure (no DOM): the verification script
// drives LatchRun directly. The rendering half mounts seven canvases.
//
// Widget conventions (docs/conventions/viz-style.md): theme colors from CSS
// variables, repaint on `themechange`, symbols encoded by shape AND color,
// correct/wrong marked by glyph and word, respect prefers-reduced-motion (no
// autoplay; Step advances 25 updates), deterministic seed + a reset control.

// ── Tunables ─────────────────────────────────────────────────────────────────
export const SEED = 3;           // weight draws (all four networks share it)
export const DATA_SEED = 11;     // training batches
export const TEST_SEED = 12;     // the fixed test set
export const PROFILE_SEED = 13;  // gradient-through-time inputs and directions
export const TAPE_SEED = 14;     // the one sequence drawn on the tape
export const NS = 4;             // signals A B C D  = symbols 0..3
export const ND = 4;             // distractors w x y z = symbols 4..7
export const V = NS + ND;        // input alphabet (one-hot)
export const NH = 16;            // hidden units / cells
export const BATCH = 32;         // sequences per update
export const LR = 0.01;          // Adam
export const BETA1 = 0.9;
export const BETA2 = 0.999;
export const ADAM_EPS = 1e-8;
export const CLIP = 1.0;         // global gradient-norm clip (Pascanu et al. 2013)
export const MAX_STEPS = 1500;   // the run stops here (curves stay on screen)
export const EVAL_EVERY = 25;    // test accuracy / gradient profile refresh
export const TEST_M = 64;        // test set: 64 sequences of the current length
export const PROFILE_M = 64;     // gradient profile: 64 sequences of the current length
export const SOLVED_LOSS = 0.05; // "solved": mean batch loss of the last 20 updates below this
export const SOLVED_WINDOW = 20;
export const LAG_CHOICES = [10, 20, 40, 80, 160] as const;
export const DEFAULT_LAG = 80;
const STEPS_PER_TICK = 2;        // at most this many updates per animation frame
const TICK_BUDGET_MS = 30;       // ...and stop early once a frame's updates took this long

export type NetId = 'rnn' | 'forget1' | 'open' | 'carousel';
export interface NetSpec {
  id: NetId;
  label: string;
  kind: 'rnn' | 'lstm';
  /** Input-gate and forget-gate bias (LSTM only). */
  biasI: number;
  biasF: number;
  /** Train with the 1997 cell-only gradient. */
  cellOnly: boolean;
}
export const NETS: readonly NetSpec[] = [
  { id: 'rnn', label: 'RNN', kind: 'rnn', biasI: 0, biasF: 0, cellOnly: false },
  { id: 'forget1', label: 'LSTM · forget bias 1', kind: 'lstm', biasI: 0, biasF: 1, cellOnly: false },
  { id: 'open', label: 'LSTM · gates open', kind: 'lstm', biasI: -5, biasF: 5, cellOnly: false },
  { id: 'carousel', label: 'LSTM · carousel only', kind: 'lstm', biasI: -5, biasF: 5, cellOnly: true },
];
export const DEFAULT_NET: NetId = 'open';
export const netSpec = (id: NetId): NetSpec => NETS.find((n) => n.id === id) ?? NETS[2];

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

/** T x m latch sequences, time-major: symbol of sequence i at step t is S[t*m + i].
 * Step 0 is a signal (0..3); every later step a distractor (4..7). The target
 * of sequence i is S[i]. */
export function makeLatch(T: number, m: number, rand: () => number): Int32Array {
  const S = new Int32Array(T * m);
  for (let i = 0; i < m; i++) {
    S[i] = Math.floor(rand() * NS);
    for (let t = 1; t < T; t++) S[t * m + i] = NS + Math.floor(rand() * ND);
  }
  return S;
}

/** Xavier normal: std = sqrt(1 / fan_in), as nanograd's xavier_normal. */
function xavier(nIn: number, nOut: number, rand: () => number): Float64Array {
  const W = new Float64Array(nIn * nOut);
  const std = Math.sqrt(1 / nIn);
  for (let i = 0; i < W.length; i++) W[i] = std * gauss(rand);
  return W;
}

/** Overflow-safe logistic sigmoid (e^-|z| in both branches, as nanograd). */
function sigmoid(z: number): number {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

// ── The network: RNN or LSTM (16 units) + Linear(16 -> 4) on the last state ──

/** Everything a forward pass keeps for backward (and for drawing). Row (t, i)
 * of H, C, TC starts at (t*m + i)*NH; of G (activated gates i f g o) at
 * (t*m + i)*4*NH. C, TC, G are null for the RNN. */
export interface Cache {
  H: Float64Array;
  C: Float64Array | null;
  TC: Float64Array | null;
  G: Float64Array | null;
}

/** Loss and gradients of one batch, in params() order. */
interface BatchResult { loss: number; grads: Float64Array[] }

/** What the profile returns: batch-mean norms, relative to k = 0, for
 * k = 0 .. T-1 steps back from the last step. `c` is null for the RNN. */
export interface Profile { x: Float64Array; c: Float64Array | null }

export class LatchNet {
  readonly spec: NetSpec;
  /** Width of the recurrent layer's pre-activation: NH (RNN) or 4*NH (LSTM). */
  readonly GN: number;
  /** Win: V x GN; Wrec: NH x GN; Why: NH x NS (row-major, rows = fan-in). */
  Win: Float64Array; Wrec: Float64Array; b: Float64Array;
  Why: Float64Array; bo = new Float64Array(NS);

  /** Draw order: input weights, recurrent weights, read-out from one stream;
   * biases zero except the LSTM gate biases. */
  constructor(spec: NetSpec, seed: number) {
    this.spec = spec;
    this.GN = spec.kind === 'lstm' ? 4 * NH : NH;
    const rand = mulberry32(seed);
    this.Win = xavier(V, this.GN, rand);
    this.Wrec = xavier(NH, this.GN, rand);
    this.Why = xavier(NH, NS, rand);
    this.b = new Float64Array(this.GN);
    if (spec.kind === 'lstm') {
      this.b.fill(spec.biasI, 0, NH);
      this.b.fill(spec.biasF, NH, 2 * NH);
    }
  }

  /** Every parameter array, canonical order (the gradient list matches). */
  params(): Float64Array[] {
    return [this.Win, this.Wrec, this.b, this.Why, this.bo];
  }

  /** Forward through time. S: T x m symbols (time-major). h<-1> = c<-1> = 0.
   * One-hot x Win is just a row lookup of Win. `frozenH`, if given, is what the
   * gates read as h<t-1> (a constant copy: used only to check the cell-only
   * gradient, which is the exact gradient of that modified network). */
  forward(S: Int32Array, T: number, m: number, frozenH?: Float64Array): Cache {
    const { GN } = this;
    const lstm = this.spec.kind === 'lstm';
    const H = new Float64Array(T * m * NH);
    const C = lstm ? new Float64Array(T * m * NH) : null;
    const TC = lstm ? new Float64Array(T * m * NH) : null;
    const G = lstm ? new Float64Array(T * m * GN) : null;
    const z = new Float64Array(GN);
    const hs = frozenH ?? H;
    for (let t = 0; t < T; t++)
      for (let i = 0; i < m; i++) {
        const p = t * m + i;
        const xo = S[p] * GN;
        for (let j = 0; j < GN; j++) z[j] = this.Win[xo + j] + this.b[j];
        if (t > 0) {
          const po = ((t - 1) * m + i) * NH;
          for (let k = 0; k < NH; k++) {
            const h = hs[po + k];
            const wo = k * GN;
            for (let j = 0; j < GN; j++) z[j] += h * this.Wrec[wo + j];
          }
        }
        const ho = p * NH;
        if (!lstm) {
          for (let j = 0; j < NH; j++) H[ho + j] = Math.tanh(z[j]);
          continue;
        }
        const go = p * GN;
        const po = ((t - 1) * m + i) * NH;
        for (let j = 0; j < NH; j++) {
          const ig = sigmoid(z[j]);
          const fg = sigmoid(z[NH + j]);
          const gg = Math.tanh(z[2 * NH + j]);
          const og = sigmoid(z[3 * NH + j]);
          G![go + j] = ig; G![go + NH + j] = fg; G![go + 2 * NH + j] = gg; G![go + 3 * NH + j] = og;
          const cPrev = t > 0 ? C![po + j] : 0;
          const c = fg * cPrev + ig * gg;      // keep some, write some
          const tc = Math.tanh(c);
          C![ho + j] = c; TC![ho + j] = tc;
          H[ho + j] = og * tc;                 // show some
        }
      }
    return { H, C, TC, G };
  }

  /** Read-out logits of sequence i at the last step. */
  private logitsLast(H: Float64Array, T: number, m: number, i: number, out: Float64Array): void {
    const o = ((T - 1) * m + i) * NH;
    for (let c = 0; c < NS; c++) {
      let s = this.bo[c];
      for (let k = 0; k < NH; k++) s += H[o + k] * this.Why[k * NS + c];
      out[c] = s;
    }
  }

  /** Argmax read-out at the last step, one answer per sequence. */
  predictLast(H: Float64Array, T: number, m: number): Uint8Array {
    const pred = new Uint8Array(m);
    const lg = new Float64Array(NS);
    for (let i = 0; i < m; i++) {
      this.logitsLast(H, T, m, i, lg);
      let best = 0;
      for (let c = 1; c < NS; c++) if (lg[c] > lg[best]) best = c;
      pred[i] = best;
    }
    return pred;
  }

  /** Backpropagation through time, written out by hand, from dHlast (m x NH):
   * the gradient into the last hidden state, the only place the loss touches
   * the recurrent layer. Returns the gradients of Win, Wrec, b. With
   * `keep`, also every step's pre-activation gradient dZ (T*m*GN) and, for the
   * LSTM, the cell's gradient through time dC (T*m*NH). */
  backprop(S: Int32Array, T: number, m: number, cache: Cache, dHlast: Float64Array,
           cellOnly: boolean, keep: boolean) {
    const { GN } = this;
    const lstm = this.spec.kind === 'lstm';
    const { H, C, TC, G } = cache;
    const gWin = new Float64Array(V * GN);
    const gWrec = new Float64Array(NH * GN);
    const gb = new Float64Array(GN);
    const dZ = keep ? new Float64Array(T * m * GN) : null;
    const dC = keep && lstm ? new Float64Array(T * m * NH) : null;
    const dhNext = new Float64Array(m * NH);
    const dcNext = new Float64Array(m * NH);
    const dz = new Float64Array(GN);
    for (let t = T - 1; t >= 0; t--)
      for (let i = 0; i < m; i++) {
        const p = t * m + i;
        const ho = p * NH;
        const no = i * NH;
        const pv = ((t - 1) * m + i) * NH;
        if (!lstm) {
          for (let j = 0; j < NH; j++) {
            const h = H[ho + j];
            const dh = dhNext[no + j] + (t === T - 1 ? dHlast[no + j] : 0);
            dz[j] = dh * (1 - h * h);
          }
        } else {
          const go = p * GN;
          for (let j = 0; j < NH; j++) {
            const dh = dhNext[no + j] + (t === T - 1 ? dHlast[no + j] : 0);
            const ig = G![go + j], fg = G![go + NH + j];
            const gg = G![go + 2 * NH + j], og = G![go + 3 * NH + j];
            const tc = TC![ho + j];
            // Along the carousel (dcNext) and via h<t> = o tanh(c<t>).
            const dc = dcNext[no + j] + dh * og * (1 - tc * tc);
            if (dC) dC[ho + j] = dc;
            const cPrev = t > 0 ? C![pv + j] : 0;
            dz[j] = dc * gg * (ig * (1 - ig));
            dz[NH + j] = dc * cPrev * (fg * (1 - fg));
            dz[2 * NH + j] = dc * ig * (1 - gg * gg);
            dz[3 * NH + j] = dh * tc * (og * (1 - og));
            dcNext[no + j] = dc * fg;           // one step back along the cell
          }
        }
        const xo = S[p] * GN;
        for (let j = 0; j < GN; j++) { gWin[xo + j] += dz[j]; gb[j] += dz[j]; }
        if (t > 0)
          for (let k = 0; k < NH; k++) {
            const hp = H[pv + k];
            const wo = k * GN;
            for (let j = 0; j < GN; j++) gWrec[wo + j] += hp * dz[j];
          }
        if (dZ) dZ.set(dz, p * GN);
        // THE POINT OF THE MODULE. Through the gates, one step back multiplies
        // by Wrec^T (and the squashing slopes); the cell path above multiplies
        // by f alone. Cell-only training declines to take the first step.
        if (cellOnly) {
          for (let k = 0; k < NH; k++) dhNext[no + k] = 0;
        } else {
          for (let k = 0; k < NH; k++) {
            let s = 0;
            const wo = k * GN;
            for (let j = 0; j < GN; j++) s += dz[j] * this.Wrec[wo + j];
            dhNext[no + k] = s;
          }
        }
      }
    return { gWin, gWrec, gb, dZ, dC };
  }

  /** Latch loss and gradients on one batch: forward, softmax cross-entropy of
   * the last step's read-out against the first symbol (mean over the batch),
   * then BPTT. dH is zero at every step but the last. Does not touch the
   * weights. */
  lossAndGrads(S: Int32Array, T: number, m: number,
               cellOnly = this.spec.cellOnly, frozenH?: Float64Array): BatchResult {
    const cache = this.forward(S, T, m, frozenH);
    const dHlast = new Float64Array(m * NH);
    const gWhy = new Float64Array(NH * NS);
    const gbo = new Float64Array(NS);
    const lg = new Float64Array(NS);
    let total = 0;
    for (let i = 0; i < m; i++) {
      this.logitsLast(cache.H, T, m, i, lg);
      let mx = -Infinity;
      for (let c = 0; c < NS; c++) if (lg[c] > mx) mx = lg[c];
      let sum = 0;
      for (let c = 0; c < NS; c++) { lg[c] = Math.exp(lg[c] - mx); sum += lg[c]; }
      const y = S[i];
      total += -Math.log(lg[y] / sum);
      const o = ((T - 1) * m + i) * NH;
      for (let c = 0; c < NS; c++) {
        const d = (lg[c] / sum - (c === y ? 1 : 0)) / m;      // dL/dlogit = (P - onehot)/m
        gbo[c] += d;
        for (let k = 0; k < NH; k++) {
          gWhy[k * NS + c] += cache.H[o + k] * d;
          dHlast[i * NH + k] += d * this.Why[k * NS + c];
        }
      }
    }
    const { gWin, gWrec, gb } = this.backprop(S, T, m, cache, dHlast, cellOnly, false);
    return { loss: total / m, grads: [gWin, gWrec, gb, gWhy, gbo] };
  }

  /** Accuracy of the last-step answer over a sequence set. */
  accuracy(S: Int32Array, T: number, m: number): number {
    const pred = this.predictLast(this.forward(S, T, m).H, T, m);
    let ok = 0;
    for (let i = 0; i < m; i++) if (pred[i] === S[i]) ok++;
    return ok / m;
  }

  /** How much of the gradient at the LAST step reaches each earlier step
   * (grad_through_time in the Python). Send a unit-norm random gradient into the
   * final hidden state of every sequence, walk back through time, and return
   * the batch mean of ||dL/dx<T-1-k>|| (the gradient reaching the input k steps
   * back -- comparable across the RNN and the LSTM) and, for an LSTM,
   * ||dL/dc<T-1-k>|| (the carousel), each relative to k = 0, for k = 0 .. T-1.
   * The walk uses the network's own training rule (cell-only for the carousel
   * network), so the bars show the gradient the training actually sees. */
  gradProfile(S: Int32Array, T: number, m: number, g: Float64Array): Profile {
    const cache = this.forward(S, T, m);
    const dHlast = new Float64Array(m * NH);
    for (let i = 0; i < m; i++) {
      let nrm = 0;
      for (let k = 0; k < NH; k++) nrm += g[i * NH + k] * g[i * NH + k];
      nrm = Math.sqrt(nrm);
      for (let k = 0; k < NH; k++) dHlast[i * NH + k] = g[i * NH + k] / nrm;
    }
    const { dZ, dC } = this.backprop(S, T, m, cache, dHlast, this.spec.cellOnly, true);
    const { GN } = this;
    const x = new Float64Array(T);
    const c = dC ? new Float64Array(T) : null;
    for (let t = 0; t < T; t++) {
      const k = T - 1 - t;
      for (let i = 0; i < m; i++) {
        const p = t * m + i;
        let n2 = 0;
        for (let v = 0; v < V; v++) {                  // dX = dz Win^T
          let s = 0;
          for (let j = 0; j < GN; j++) s += this.Win[v * GN + j] * dZ![p * GN + j];
          n2 += s * s;
        }
        x[k] += Math.sqrt(n2) / m;
        if (c && dC) {
          let c2 = 0;
          for (let j = 0; j < NH; j++) c2 += dC[p * NH + j] * dC[p * NH + j];
          c[k] += Math.sqrt(c2) / m;
        }
      }
    }
    const x0 = x[0] > 0 ? x[0] : 1;
    for (let k = 0; k < T; k++) x[k] /= x0;
    if (c) {
      const c0 = c[0] > 0 ? c[0] : 1;
      for (let k = 0; k < T; k++) c[k] /= c0;
    }
    return { x, c };
  }

  /** Mean forget gate over steps 1.. of sequence 0 of a cache (m = 1 for the
   * tape), or null for the RNN. */
  meanForget(cache: Cache, T: number): number | null {
    if (!cache.G) return null;
    const { GN } = this;
    let s = 0, n = 0;
    for (let t = 1; t < T; t++)
      for (let j = 0; j < NH; j++) { s += cache.G[t * GN + NH + j]; n++; }
    return n > 0 ? s / n : null;
  }
}

/** Unit-direction source for the gradient profile: m x NH Gaussian draws. */
function profileDirections(m: number, rand: () => number): Float64Array {
  const g = new Float64Array(m * NH);
  for (let j = 0; j < g.length; j++) g[j] = gauss(rand);
  return g;
}

/** One training run: network + Adam + data stream + fixed evaluation sets. */
export class LatchRun {
  readonly spec: NetSpec;
  readonly lag: number;
  readonly net: LatchNet;
  steps = 0;
  /** Training loss and pre-clip gradient norm of every update so far. */
  readonly lossHist: number[] = [];
  readonly normHist: number[] = [];
  /** Training loss of the most recent update. */
  loss = Math.log(NS);
  /** Gradient norm before clipping, most recent update. */
  gradNorm = 0;
  /** Number of updates whose gradient was clipped; the largest norm seen. */
  clipCount = 0;
  maxNorm = 0;
  /** First update whose trailing 20-update mean batch loss is below 0.05. */
  solvedAt: number | null = null;

  private readonly dataRand = mulberry32(DATA_SEED);
  private readonly testS: Int32Array;
  private readonly profS: Int32Array;
  private readonly profG: Float64Array;
  private readonly mom: Float64Array[];
  private readonly vel: Float64Array[];

  constructor(spec: NetSpec, lag: number, seed = SEED) {
    this.spec = spec;
    this.lag = lag;
    this.net = new LatchNet(spec, seed);
    this.testS = makeLatch(lag, TEST_M, mulberry32(TEST_SEED));
    const pr = mulberry32(PROFILE_SEED);
    this.profS = makeLatch(lag, PROFILE_M, pr);
    this.profG = profileDirections(PROFILE_M, pr);
    this.mom = this.net.params().map((p) => new Float64Array(p.length));
    this.vel = this.net.params().map((p) => new Float64Array(p.length));
  }

  /** One update on a fresh random batch: gradients, global-norm clip, Adam. */
  update(): void {
    const S = makeLatch(this.lag, BATCH, this.dataRand);
    const { loss, grads } = this.net.lossAndGrads(S, this.lag, BATCH);
    this.loss = loss;

    // Clip: treat every gradient as one vector g; if ||g|| > 1, g *= 1/||g||.
    let total = 0;
    for (const g of grads) for (let j = 0; j < g.length; j++) total += g[j] * g[j];
    const norm = Math.sqrt(total);
    this.gradNorm = norm;
    if (norm > this.maxNorm) this.maxNorm = norm;
    if (norm > CLIP) {
      this.clipCount++;
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

    this.lossHist.push(loss);
    this.normHist.push(norm);
    if (this.solvedAt === null && this.steps >= SOLVED_WINDOW) {
      let s = 0;
      for (let k = this.lossHist.length - SOLVED_WINDOW; k < this.lossHist.length; k++) s += this.lossHist[k];
      if (s / SOLVED_WINDOW < SOLVED_LOSS) this.solvedAt = this.steps;
    }
  }

  /** Test accuracy on the fixed 64 sequences of length `lag`. */
  evaluate(): number {
    return this.net.accuracy(this.testS, this.lag, TEST_M);
  }

  /** Relative ||dL/dx|| (and ||dL/dc||) k = 0 .. lag-1 steps back, fixed batch. */
  profile(): Profile {
    return this.net.gradProfile(this.profS, this.lag, PROFILE_M, this.profG);
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
const NORM_MIN = 1e-3, NORM_MAX = 1e2;
const PROF_MIN = 1e-8, PROF_MAX = 1e1;
const MAX_BARS = 80;                 // gradient profile: at most this many bars/buckets
const LEFT = 64;                     // tape, memory and gate panels share their column grid
const SHAPES = ['circle', 'square', 'triangle', 'diamond'] as const;
const SIGNAL_NAMES = ['A', 'B', 'C', 'D'] as const;
const MEM_SCALE = { rnn: 1, lstm: 2 } as const;   // |value| that fills a heatmap cell

export interface LatchStats {
  step: number;
  net: NetId;
  label: string;
  lag: number;
  acc: number;
  loss: number;
  gradNorm: number;
  clipped: number;
  solvedAt: number | null;
  /** Mean forget gate on the tape sequence; null for the RNN. */
  meanForget: number | null;
  playing: boolean;
  done: boolean;
}

export interface LatchHandles {
  toggle(): void;
  step(): void;
  reset(): void;
  setNet(id: NetId): void;
  setLag(T: number): void;
  destroy(): void;
  reducedMotion: boolean;
}

export interface LatchCanvases {
  tape: HTMLCanvasElement;
  memory: HTMLCanvasElement;
  gates: HTMLCanvasElement;
  acc: HTMLCanvasElement;
  loss: HTMLCanvasElement;
  norm: HTMLCanvasElement;
  grad: HTMLCanvasElement;
}

export function mountLatch(
  cv: LatchCanvases,
  opts: { onStats?: (s: LatchStats) => void; net?: NetId; lag?: number } = {},
): LatchHandles {
  const ctx2 = (c: HTMLCanvasElement) => {
    const g = c.getContext('2d');
    if (!g) throw new Error('2D canvas context unavailable');
    return g;
  };
  const cTape = ctx2(cv.tape), cMem = ctx2(cv.memory), cGates = ctx2(cv.gates);
  const cAcc = ctx2(cv.acc), cLoss = ctx2(cv.loss), cNorm = ctx2(cv.norm), cGrad = ctx2(cv.grad);

  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let palette = readPalette();
  let spec: NetSpec = netSpec(opts.net ?? DEFAULT_NET);
  let lag: number = opts.lag ?? DEFAULT_LAG;
  let playing = false;
  let raf = 0;

  // One fixed sequence for the tape, memory and gate panels.
  let tapeS = makeLatch(lag, 1, mulberry32(TAPE_SEED));

  let run: LatchRun;
  let curve: { step: number; acc: number }[] = [];
  let evalStep = -1;
  let acc = 0;
  let prof: Profile = { x: new Float64Array(lag), c: null };
  let meanF: number | null = null;

  function restart() {
    run = new LatchRun(spec, lag);
    tapeS = makeLatch(lag, 1, mulberry32(TAPE_SEED));
    curve = [];
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

  /** One signal symbol: a distinct shape AND a distinct color. */
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

  /** Width of one time step on the shared column grid. */
  const colW = (c: HTMLCanvasElement) => (c.width - 8 - LEFT) / lag;

  function drawTape(answer: number) {
    const { width } = cv.tape;
    const g = cTape;
    clear(g, cv.tape);
    const w = colW(cv.tape);
    const sig = tapeS[0];
    const rows = { sig: 38, strip: 56, ans: 112 };
    const laneH = 5;
    text(g, `the tape — one test sequence, T = ${lag}`, 8, 14, palette.text);

    // The signal: big, shape + color + letter.
    text(g, 'signal', LEFT - 8, rows.sig + 4, palette.muted, 'right');
    glyph(g, sig, LEFT + 12, rows.sig, 9);
    text(g, `${SIGNAL_NAMES[sig]}  at t = 0`, LEFT + 28, rows.sig + 4, palette.text);

    // The distractors: a strip with one tick per step; the symbol is the tick's
    // lane (w highest .. z lowest), so it reads at any T without color.
    text(g, 'distractors', LEFT - 8, rows.strip + 2 * laneH + 4, palette.muted, 'right');
    g.strokeStyle = palette.axis;
    g.lineWidth = 1;
    g.globalAlpha = 0.5;
    g.strokeRect(LEFT + w, rows.strip - 2, (lag - 1) * w, ND * laneH + 4);
    g.globalAlpha = 1;
    g.fillStyle = palette.muted;
    const tw = Math.max(1, Math.min(w * 0.6, 5));
    for (let t = 1; t < lag; t++) {
      const lane = tapeS[t] - NS;
      g.fillRect(LEFT + (t + 0.5) * w - tw / 2, rows.strip + lane * laneH, tw, laneH - 1);
    }
    text(g, `w x y z = top to bottom lane · ${lag - 1} steps`, LEFT + w, rows.strip + ND * laneH + 18);
    text(g, `t = ${lag - 1}`, width - 8, rows.strip + ND * laneH + 18, palette.muted, 'right');

    // The answer at the last step, against the target.
    const ok = answer === sig;
    text(g, `at t = ${lag - 1}`, LEFT - 8, rows.ans + 4, palette.muted, 'right');
    text(g, 'target', LEFT, rows.ans + 4);
    glyph(g, sig, LEFT + 52, rows.ans, 8);
    text(g, 'answer', LEFT + 82, rows.ans + 4);
    glyph(g, answer, LEFT + 146, rows.ans, 8);
    if (ok) {
      text(g, '✓ correct', LEFT + 166, rows.ans + 4, palette.text);
    } else {                                   // wrong: ring + cross + word, not color alone
      g.strokeStyle = palette.action;
      g.lineWidth = 2;
      g.beginPath(); g.arc(LEFT + 146, rows.ans, 13, 0, Math.PI * 2); g.stroke();
      text(g, '✗ wrong', LEFT + 166, rows.ans + 4, palette.action);
    }
  }

  /** Heatmap of 16 rows x lag columns with a diverging palette (blank at zero). */
  function drawMemory(cache: Cache) {
    const g = cMem;
    clear(g, cv.memory);
    const lstm = spec.kind === 'lstm';
    const M = lstm ? cache.C! : cache.H;
    const scale = lstm ? MEM_SCALE.lstm : MEM_SCALE.rnn;
    const w = colW(cv.memory);
    const top = 22, ch = 6;
    text(g, lstm ? `cell state c (${NH} cells × ${lag} steps)` : `hidden state h (${NH} units × ${lag} steps)`,
      8, 14, palette.text);
    for (let u = 0; u < NH; u++)
      for (let t = 0; t < lag; t++) {
        const v = M[t * NH + u];
        g.globalAlpha = Math.min(1, Math.abs(v) / scale);
        g.fillStyle = v >= 0 ? palette.pos : palette.neg;
        g.fillRect(LEFT + t * w, top + u * ch, Math.ceil(w), ch);
      }
    g.globalAlpha = 1;
    g.strokeStyle = palette.axis;
    g.lineWidth = 1;
    g.strokeRect(LEFT, top, lag * w, NH * ch);
    text(g, lstm ? 'cell 1' : 'unit 1', LEFT - 6, top + 9, palette.muted, 'right');
    text(g, lstm ? 'cell 16' : 'unit 16', LEFT - 6, top + NH * ch - 1, palette.muted, 'right');
    const ly = top + NH * ch + 16;
    g.fillStyle = palette.pos; g.fillRect(LEFT, ly - 8, 10, 8);
    text(g, `+${scale}`, LEFT + 14, ly, palette.muted);
    g.fillStyle = palette.neg; g.fillRect(LEFT + 44, ly - 8, 10, 8);
    text(g, `−${scale}`, LEFT + 58, ly, palette.muted);
    text(g, '0 = blank', LEFT + 90, ly, palette.muted);
    text(g, `t = ${lag - 1}`, cv.memory.width - 8, ly, palette.muted, 'right');
  }

  /** Input-gate and forget-gate strips (LSTM), or a note (RNN). */
  function drawGates(cache: Cache) {
    const g = cGates;
    clear(g, cv.gates);
    if (spec.kind !== 'lstm' || !cache.G) {
      text(g, 'gates', 8, 14, palette.text);
      text(g, 'no gates — the RNN rewrites its whole state', LEFT, 70);
      text(g, 'at every step, so there is nothing to keep open.', LEFT, 88);
      return;
    }
    const w = colW(cv.gates);
    const ch = 4, GN = 4 * NH;
    const strip = (label: string, off: number, top: number) => {
      text(g, label, 8, top - 6, palette.text);
      for (let u = 0; u < NH; u++)
        for (let t = 0; t < lag; t++) {
          g.globalAlpha = Math.min(1, Math.max(0, cache.G![t * GN + off * NH + u]));
          g.fillStyle = palette.dataA;
          g.fillRect(LEFT + t * w, top + u * ch, Math.ceil(w), ch);
        }
      g.globalAlpha = 1;
      g.strokeStyle = palette.axis;
      g.lineWidth = 1;
      g.strokeRect(LEFT, top, lag * w, NH * ch);
      text(g, 'cell 1', LEFT - 6, top + 8, palette.muted, 'right');
      text(g, 'cell 16', LEFT - 6, top + NH * ch - 1, palette.muted, 'right');
    };
    strip('input gate i — how much to write', 0, 26);
    strip('forget gate f — how much of the old cell to keep', 1, 26 + NH * ch + 28);
    const ly = 26 + 2 * NH * ch + 28 + 20;
    // A 0..1 ramp: darker = more open; the end labels give the numbers.
    const rw = 90;
    for (let k = 0; k < rw; k++) {
      g.globalAlpha = k / (rw - 1);
      g.fillStyle = palette.dataA;
      g.fillRect(LEFT + k, ly - 8, 1, 8);
    }
    g.globalAlpha = 1;
    g.strokeStyle = palette.axis;
    g.strokeRect(LEFT, ly - 8, rw, 8);
    text(g, '0 = shut (blank)', LEFT + rw + 8, ly, palette.muted);
    text(g, '1 = open', cv.gates.width - 8, ly, palette.muted, 'right');
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

  /** Light horizontal gridline with a y label. */
  function grid(g: CanvasRenderingContext2D, left: number, right: number, y: number, label: string) {
    text(g, label, left - 4, y + 4, palette.muted, 'right');
    g.globalAlpha = 0.25;
    g.strokeStyle = palette.axis;
    g.lineWidth = 1;
    g.beginPath(); g.moveTo(left, y); g.lineTo(right, y); g.stroke();
    g.globalAlpha = 1;
  }

  function drawAcc() {
    const g = cAcc;
    const { left, right, top, bottom } = frame(g, cv.acc, 'test accuracy (64 sequences, last-step answer)', 24);
    const yOf = (v: number) => bottom - v * (bottom - top);
    const xOf = (k: number) => left + (k / MAX_STEPS) * (right - left);
    for (const v of [0, 0.5, 1]) grid(g, left, right, yOf(v), `${Math.round(v * 100)}%`);
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
    const { left, right, top, bottom } = frame(g, cv.loss, 'training loss (log scale, every update)', 24);
    const yOf = (v: number) => {
      const t = (L10(Math.min(LOSS_MAX, Math.max(LOSS_MIN, v))) - L10(LOSS_MIN)) / (L10(LOSS_MAX) - L10(LOSS_MIN));
      return bottom - t * (bottom - top);
    };
    const xOf = (k: number) => left + (k / MAX_STEPS) * (right - left);
    for (const v of [1e-3, 1e-2, 1e-1, 1]) grid(g, left, right, yOf(v), v >= 1 ? '1' : `1e${Math.round(L10(v))}`);
    g.setLineDash([2, 3]);              // ln 4: loss of guessing uniformly
    g.strokeStyle = palette.muted;
    g.beginPath(); g.moveTo(left, yOf(Math.log(NS))); g.lineTo(right, yOf(Math.log(NS))); g.stroke();
    g.setLineDash([]);
    text(g, 'ln 4', right, yOf(Math.log(NS)) - 4, palette.muted, 'right');
    text(g, '0', left, cv.loss.height - 8);
    text(g, `${MAX_STEPS} updates`, right, cv.loss.height - 8, palette.muted, 'right');
    g.strokeStyle = palette.text;
    g.lineWidth = 1.5;
    g.beginPath();
    run.lossHist.forEach((v, k) => { if (k === 0) g.moveTo(xOf(1), yOf(v)); else g.lineTo(xOf(k + 1), yOf(v)); });
    g.stroke();
  }

  function drawNorm() {
    const g = cNorm;
    const { left, right, top, bottom } = frame(
      g, cv.norm, `gradient norm before clipping (log) · clipped ${run.clipCount}×`, 24);
    const yOf = (v: number) => {
      const t = (L10(Math.min(NORM_MAX, Math.max(NORM_MIN, v))) - L10(NORM_MIN)) / (L10(NORM_MAX) - L10(NORM_MIN));
      return bottom - t * (bottom - top);
    };
    const xOf = (k: number) => left + (k / MAX_STEPS) * (right - left);
    for (const e of [-2, 0, 2]) grid(g, left, right, yOf(10 ** e), e === 0 ? '1' : `1e${e}`);
    g.strokeStyle = palette.action;     // the clip threshold: dashed, labelled
    g.lineWidth = 1.5;
    g.setLineDash([5, 3]);
    g.beginPath(); g.moveTo(left, yOf(CLIP)); g.lineTo(right, yOf(CLIP)); g.stroke();
    g.setLineDash([]);
    text(g, 'clip 1.0', right, yOf(CLIP) - 4, palette.action, 'right');
    text(g, '0', left, cv.norm.height - 8);
    text(g, `${MAX_STEPS} updates`, right, cv.norm.height - 8, palette.muted, 'right');
    g.strokeStyle = palette.dataA;
    g.lineWidth = 1.2;
    g.beginPath();
    run.normHist.forEach((v, k) => { if (k === 0) g.moveTo(xOf(1), yOf(v)); else g.lineTo(xOf(k + 1), yOf(v)); });
    g.stroke();
    g.fillStyle = palette.action;       // clipped updates: a dot above the line
    run.normHist.forEach((v, k) => {
      if (v <= CLIP) return;
      g.beginPath(); g.arc(xOf(k + 1), yOf(v), 2, 0, Math.PI * 2); g.fill();
    });
  }

  /** Geometric mean of v[a..b) -- one bar for a bucket of steps. */
  function bucket(v: Float64Array, a: number, b: number): number {
    let s = 0;
    for (let k = a; k < b; k++) s += Math.log(Math.max(v[k], 1e-300));
    return Math.exp(s / (b - a));
  }

  function drawGrad() {
    const g = cGrad;
    const lstm = spec.kind === 'lstm';
    const { left, right, top, bottom } = frame(
      g, cv.grad, 'gradient reaching k steps back from the last step', 44);
    const yOf = (v: number) => {
      const t = (L10(Math.min(PROF_MAX, Math.max(PROF_MIN, v))) - L10(PROF_MIN)) / (L10(PROF_MAX) - L10(PROF_MIN));
      return bottom - t * (bottom - top);
    };
    for (let e = -8; e <= 0; e += 2) grid(g, left, right, yOf(10 ** e), e === 0 ? '1' : `1e${e}`);
    const nb = Math.min(lag, MAX_BARS);
    const slot = (right - left) / nb;
    const bw = Math.max(1.5, slot * 0.7);
    const lo = (b: number) => Math.floor((b * lag) / nb);
    const hi = (b: number) => Math.floor(((b + 1) * lag) / nb);
    g.fillStyle = palette.dataA;
    for (let b = 0; b < nb; b++) {
      const y = yOf(bucket(prof.x, lo(b), hi(b)));
      g.fillRect(left + b * slot + (slot - bw) / 2, y, bw, bottom - y);
    }
    if (lstm && prof.c) {              // the cell path: a line with round markers
      g.strokeStyle = palette.accent;
      g.fillStyle = palette.accent;
      g.lineWidth = 2;
      g.beginPath();
      for (let b = 0; b < nb; b++) {
        const x = left + (b + 0.5) * slot, y = yOf(bucket(prof.c, lo(b), hi(b)));
        if (b === 0) g.moveTo(x, y); else g.lineTo(x, y);
      }
      g.stroke();
      if (nb <= 40)
        for (let b = 0; b < nb; b++) {
          g.beginPath();
          g.arc(left + (b + 0.5) * slot, yOf(bucket(prof.c, lo(b), hi(b))), 2.2, 0, Math.PI * 2);
          g.fill();
        }
    }
    // k = T-1: where the signal is. Solid vertical line, labelled.
    const sx = left + (nb - 0.5) * slot;
    g.strokeStyle = palette.action;
    g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(sx, top); g.lineTo(sx, bottom); g.stroke();
    text(g, `signal: k = ${lag - 1}`, sx - 4, top + 11, palette.action, 'right');
    text(g, '0', left + 0.5 * slot, bottom + 14, palette.muted, 'center');
    text(g, `${Math.round((lag - 1) / 2)}`, (left + right) / 2, bottom + 14, palette.muted, 'center');
    text(g, `${lag - 1}`, sx, bottom + 14, palette.muted, 'center');
    // (No x-axis label: the frame's title already says what k is.)
    // Legend (shape/line, not color alone).
    g.fillStyle = palette.dataA;
    g.fillRect(left + 4, cv.grad.height - 13, 10, 8);
    text(g, lstm ? 'bars: ‖∂L/∂x‖' : 'bars: ‖∂L/∂x‖ (input)', left + 18, cv.grad.height - 5, palette.muted);
    if (lstm) {
      g.strokeStyle = palette.accent;
      g.lineWidth = 2;
      g.beginPath(); g.moveTo(left + 110, cv.grad.height - 9); g.lineTo(left + 128, cv.grad.height - 9); g.stroke();
      text(g, 'line: ‖∂L/∂c‖ (cell)', left + 132, cv.grad.height - 5, palette.muted);
    }
  }

  function draw() {
    const cache = run.net.forward(tapeS, lag, 1);
    meanF = run.net.meanForget(cache, lag);
    drawTape(run.net.predictLast(cache.H, lag, 1)[0]);
    drawMemory(cache);
    drawGates(cache);
    drawAcc();
    drawLoss();
    drawNorm();
    drawGrad();
  }

  function emit() {
    opts.onStats?.({
      step: run.steps, net: spec.id, label: spec.label, lag, acc, loss: run.loss,
      gradNorm: run.gradNorm, clipped: run.clipCount, solvedAt: run.solvedAt,
      meanForget: meanF, playing, done: run.steps >= MAX_STEPS,
    });
  }

  function advance(n: number, budgetMs = Infinity, evalNow = true) {
    const t0 = performance.now();
    let crossed = false;
    for (let k = 0; k < n && run.steps < MAX_STEPS; k++) {
      if (k > 0 && performance.now() - t0 > budgetMs) break;
      run.update();
      if (run.steps % EVAL_EVERY === 0) crossed = true;
    }
    if (run.steps >= MAX_STEPS) playing = false;
    if (crossed || evalNow || run.steps >= MAX_STEPS) ensureEval();
    draw();
    emit();
  }

  // An update at T = 160 is ~25M flops, so a frame takes a couple of them
  // (within the time budget).
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
    step() { pause(); advance(EVAL_EVERY); },
    reset() { pause(); restart(); draw(); emit(); },
    setNet(id: NetId) { pause(); spec = netSpec(id); restart(); draw(); emit(); },
    setLag(T: number) { pause(); lag = T; restart(); draw(); emit(); },
    destroy() {
      pause();
      window.removeEventListener('themechange', onTheme);
    },
    reducedMotion,
  };
}
