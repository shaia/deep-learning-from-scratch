// Module 03's interactive: slide a 5×5 kernel over a digit and watch the
// feature map build, one dot product at a time. The kernels include the six
// filters conv1 actually learned on MNIST (baked into lenetFilters.json by
// topics/03-lenet/python/lenet.py --export-filters), so the reader can compare
// hand-designed edge detectors with the ones gradient descent found.
//
// Lifecycle follows exampleCanvas.ts: theme colors from CSS variables, repaint
// on `themechange`, respect prefers-reduced-motion (no autoplay; the map draws
// fully so nothing is lost), deterministic content + a reset control.

import baked from './lenetFilters.json';

interface Palette {
  bg: string;
  axis: string;
  text: string;
  neg: string; // diverging pair for signed feature-map values
  pos: string;
  accent: string; // the sliding window itself (the "action" color)
}

function readPalette(): Palette {
  const s = getComputedStyle(document.documentElement);
  const v = (name: string) => s.getPropertyValue(name).trim();
  return {
    bg: v('--color-surface'),
    axis: v('--viz-axis'),
    text: v('--color-text'),
    neg: v('--viz-data-a'),
    pos: v('--viz-data-b'),
    accent: v('--viz-accent'),
  };
}

const IMG = 28; // input digit
const KS = 5; // kernel
const OUT = IMG - KS + 1; // 24×24 feature map (valid, stride 1)

/** Hand-designed presets, then the six learned conv1 filters. */
const PRESETS: { name: string; k: number[][] }[] = [
  {
    name: 'vertical edge',
    k: Array.from({ length: KS }, () => [-1, -2, 0, 2, 1].map((x) => x / 8)),
  },
  {
    name: 'horizontal edge',
    k: [-1, -2, 0, 2, 1].map((x) => Array.from({ length: KS }, () => x / 8)),
  },
  {
    name: 'blur',
    k: Array.from({ length: KS }, () => Array.from({ length: KS }, () => 1 / 25)),
  },
  {
    name: 'sharpen',
    k: Array.from({ length: KS }, (_, p) =>
      Array.from({ length: KS }, (_, q) => (p === 2 && q === 2 ? 2 - 1 / 25 : -1 / 25)),
    ),
  },
  ...(baked.filters as number[][][]).map((k, i) => ({ name: `learned #${i + 1}`, k })),
];

const DIGITS = (baked.digits as number[][][]).map((d) =>
  d.map((row) => row.map((v) => v / 255)),
);

export const KERNEL_NAMES = PRESETS.map((p) => p.name);
export const N_DIGITS = DIGITS.length;

export interface ConvState {
  playing: boolean;
  sum: number; // dot product at the current position
}

export interface ConvHandles {
  setKernel: (i: number) => void;
  setDigit: (i: number) => void;
  toggle: () => void;
  reset: () => void;
  destroy: () => void;
  reducedMotion: boolean;
}

export function mountConvWidget(
  canvas: HTMLCanvasElement,
  opts: { onState?: (s: ConvState) => void } = {},
): ConvHandles {
  const maybeCtx = canvas.getContext('2d');
  if (!maybeCtx) throw new Error('2D canvas context unavailable');
  const ctx: CanvasRenderingContext2D = maybeCtx;

  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let palette = readPalette();
  let raf = 0;

  // --- geometry (canvas is 640×290) ---------------------------------------
  const CELL_IN = 6; // input pixel size
  const CELL_K = 22; // kernel cell size
  const CELL_OUT = 7; // feature-map cell size
  const TOP = 46;
  const IN_X = 22;
  const K_X = 236;
  const K_Y = TOP + 24;
  const OUT_X = 438;

  // --- state ----------------------------------------------------------------
  let kernelIdx = 0;
  let digitIdx = 0;
  let pos = 0; // current output cell in scan order: u*OUT + v
  let playing = !reducedMotion;
  let fmap: number[][] = [];
  let fmax = 1; // max |value| over the map, for the diverging colors

  function kernel(): number[][] {
    return PRESETS[kernelIdx].k;
  }
  function digit(): number[][] {
    return DIGITS[digitIdx];
  }

  /** The whole operation, precomputed: one dot product per output cell. */
  function computeMap() {
    const k = kernel();
    const img = digit();
    fmap = [];
    fmax = 1e-9;
    for (let u = 0; u < OUT; u++) {
      const row: number[] = [];
      for (let v = 0; v < OUT; v++) {
        let s = 0;
        for (let p = 0; p < KS; p++)
          for (let q = 0; q < KS; q++) s += k[p][q] * img[u + p][v + q];
        row.push(s);
        if (Math.abs(s) > fmax) fmax = Math.abs(s);
      }
      fmap.push(row);
    }
  }

  function emit() {
    const u = Math.floor(pos / OUT);
    const v = pos % OUT;
    opts.onState?.({ playing, sum: fmap[u][v] });
  }

  // --- drawing ---------------------------------------------------------------
  function label(text: string, x: number, y: number) {
    ctx.fillStyle = palette.axis;
    ctx.font = '12px ui-sans-serif, system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(text, x, y);
  }

  function draw() {
    const { width, height } = canvas;
    const u = Math.floor(pos / OUT);
    const v = pos % OUT;
    const img = digit();
    const k = kernel();

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = palette.bg;
    ctx.fillRect(0, 0, width, height);

    label(`input 28×28`, IN_X, TOP - 12);
    label(`kernel 5×5 (K)`, K_X, TOP - 12);
    label(`feature map 24×24`, OUT_X, TOP - 12);

    // Input digit (grayscale on the theme surface).
    for (let r = 0; r < IMG; r++)
      for (let c = 0; c < IMG; c++) {
        ctx.globalAlpha = img[r][c];
        ctx.fillStyle = palette.text;
        ctx.fillRect(IN_X + c * CELL_IN, TOP + r * CELL_IN, CELL_IN, CELL_IN);
      }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = palette.axis;
    ctx.lineWidth = 1;
    ctx.strokeRect(IN_X - 0.5, TOP - 0.5, IMG * CELL_IN + 1, IMG * CELL_IN + 1);

    // The receptive field the kernel is reading right now.
    ctx.strokeStyle = palette.accent;
    ctx.lineWidth = 2;
    ctx.strokeRect(IN_X + v * CELL_IN, TOP + u * CELL_IN, KS * CELL_IN, KS * CELL_IN);

    // Kernel as a signed heat grid with its 25 numbers.
    let kmax = 1e-9;
    for (const row of k) for (const x of row) kmax = Math.max(kmax, Math.abs(x));
    for (let p = 0; p < KS; p++)
      for (let q = 0; q < KS; q++) {
        const val = k[p][q];
        ctx.globalAlpha = Math.min(1, Math.abs(val) / kmax);
        ctx.fillStyle = val >= 0 ? palette.pos : palette.neg;
        ctx.fillRect(K_X + q * CELL_K, K_Y + p * CELL_K, CELL_K, CELL_K);
        ctx.globalAlpha = 1;
        ctx.fillStyle = palette.text;
        ctx.font = '9px ui-monospace, Menlo, monospace';
        ctx.textAlign = 'center';
        ctx.fillText(
          val.toFixed(1).replace('-0.0', '0.0'),
          K_X + q * CELL_K + CELL_K / 2,
          K_Y + p * CELL_K + CELL_K / 2 + 3,
        );
      }
    ctx.strokeStyle = palette.axis;
    ctx.strokeRect(K_X - 0.5, K_Y - 0.5, KS * CELL_K + 1, KS * CELL_K + 1);

    // Dot-product readout: patch · kernel at the current position.
    ctx.fillStyle = palette.text;
    ctx.font = '13px ui-monospace, Menlo, monospace';
    ctx.textAlign = 'left';
    const s = fmap[u][v];
    ctx.fillText(
      `Σ (patch × K) = ${s >= 0 ? '+' : ''}${s.toFixed(2)}`,
      K_X - 14,
      K_Y + KS * CELL_K + 26,
    );

    // Feature map, filled in scan order up to the slide position (all of it
    // under reduced motion, so a static frame still tells the whole story).
    const upto = reducedMotion ? OUT * OUT - 1 : pos;
    for (let i = 0; i <= upto; i++) {
      const uu = Math.floor(i / OUT);
      const vv = i % OUT;
      const val = fmap[uu][vv];
      ctx.globalAlpha = Math.min(1, Math.abs(val) / fmax);
      ctx.fillStyle = val >= 0 ? palette.pos : palette.neg;
      ctx.fillRect(OUT_X + vv * CELL_OUT, TOP + uu * CELL_OUT, CELL_OUT, CELL_OUT);
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = palette.axis;
    ctx.lineWidth = 1;
    ctx.strokeRect(OUT_X - 0.5, TOP - 0.5, OUT * CELL_OUT + 1, OUT * CELL_OUT + 1);

    // Current output cell.
    ctx.strokeStyle = palette.accent;
    ctx.lineWidth = 2;
    ctx.strokeRect(OUT_X + v * CELL_OUT, TOP + u * CELL_OUT, CELL_OUT, CELL_OUT);
  }

  // --- animation + interaction ------------------------------------------------
  function tick() {
    pos = (pos + 1) % (OUT * OUT);
    draw();
    emit();
    raf = requestAnimationFrame(tick);
  }

  function start() {
    cancelAnimationFrame(raf);
    if (playing && !reducedMotion) raf = requestAnimationFrame(tick);
    else draw();
    emit();
  }

  /** Hover/drag positions the kernel: input panel targets the window center,
   * feature-map panel targets the output cell directly. */
  function onPointer(ev: PointerEvent) {
    const rect = canvas.getBoundingClientRect();
    const x = ((ev.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((ev.clientY - rect.top) / rect.height) * canvas.height;
    let u = -1;
    let v = -1;
    if (x >= IN_X && x < IN_X + IMG * CELL_IN && y >= TOP && y < TOP + IMG * CELL_IN) {
      u = Math.floor((y - TOP) / CELL_IN) - 2;
      v = Math.floor((x - IN_X) / CELL_IN) - 2;
    } else if (
      x >= OUT_X &&
      x < OUT_X + OUT * CELL_OUT &&
      y >= TOP &&
      y < TOP + OUT * CELL_OUT
    ) {
      u = Math.floor((y - TOP) / CELL_OUT);
      v = Math.floor((x - OUT_X) / CELL_OUT);
    }
    if (u < -2 && v < -2) return;
    u = Math.max(0, Math.min(OUT - 1, u));
    v = Math.max(0, Math.min(OUT - 1, v));
    playing = false;
    pos = u * OUT + v;
    start();
  }
  canvas.addEventListener('pointermove', onPointer);
  canvas.addEventListener('pointerdown', onPointer);

  function onTheme() {
    palette = readPalette();
    draw();
  }
  window.addEventListener('themechange', onTheme);

  computeMap();
  start();

  return {
    setKernel(i: number) {
      kernelIdx = Math.max(0, Math.min(PRESETS.length - 1, i));
      computeMap();
      start();
    },
    setDigit(i: number) {
      digitIdx = Math.max(0, Math.min(DIGITS.length - 1, i));
      computeMap();
      start();
    },
    toggle() {
      playing = !playing;
      start();
    },
    reset() {
      kernelIdx = 0;
      digitIdx = 0;
      pos = 0;
      playing = !reducedMotion;
      computeMap();
      start();
    },
    destroy() {
      cancelAnimationFrame(raf);
      canvas.removeEventListener('pointermove', onPointer);
      canvas.removeEventListener('pointerdown', onPointer);
      window.removeEventListener('themechange', onTheme);
    },
    reducedMotion,
  };
}
