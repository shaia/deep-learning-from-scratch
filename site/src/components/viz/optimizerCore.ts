// Interactive optimizer race — vanilla TypeScript + Canvas, no framework.
// The playable companion to Module 02 (topics/02-training-toolkit). Four of the
// same update rules the C and Python code implement — SGD, momentum, RMSProp,
// Adam — descend a 2-D loss surface from a shared start. Watch plain SGD stall
// in the curved valley while the adaptive methods steer for the minimum.
//
// Follows the widget conventions in docs/conventions/viz-style.md:
//   • the loss-surface heatmap reads theme colors from CSS variables
//   • repaint on `themechange` (dispatched by BaseLayout's theme toggle)
//   • respect prefers-reduced-motion (no autoplay; stepping still works)
//   • deterministic (optimizers have no randomness) + a reset control
//   • click the surface to move the start point and re-race
//
// The four trajectory colors are a fixed colorblind-safe set (they encode four
// categories, so they stay constant across themes); each is also labelled.

// ── Loss surfaces: value, gradient, domain, and the known minimum ────────────
export type Surface = 'beale' | 'ravine';

interface SurfaceDef {
  f(x: number, y: number): number;
  grad(x: number, y: number): [number, number];
  domain: [number, number, number, number]; // xmin, xmax, ymin, ymax
  min: [number, number];
  start: [number, number];
  // Per-optimizer base learning rates, tuned so the race is instructive here.
  lr: Record<OptName, number>;
}

const SURFACES: Record<Surface, SurfaceDef> = {
  // Beale: a curved valley with a minimum at (3, 0.5). Classic optimizer test.
  beale: {
    f: (x, y) =>
      (1.5 - x + x * y) ** 2 +
      (2.25 - x + x * y * y) ** 2 +
      (2.625 - x + x * y * y * y) ** 2,
    grad: (x, y) => {
      const a = 1.5 - x + x * y;
      const b = 2.25 - x + x * y * y;
      const c = 2.625 - x + x * y * y * y;
      const dx = 2 * a * (y - 1) + 2 * b * (y * y - 1) + 2 * c * (y * y * y - 1);
      const dy = 2 * a * x + 2 * b * (2 * x * y) + 2 * c * (3 * x * y * y);
      return [dx, dy];
    },
    domain: [-1, 4, -1, 3],
    min: [3, 0.5],
    start: [-0.5, 2],
    lr: { sgd: 1e-3, momentum: 1e-3, rmsprop: 1.2e-2, adam: 2e-2 },
  },
  // Ravine: an ill-conditioned bowl (steep in y, shallow in x). SGD zig-zags;
  // momentum/RMSProp/Adam cross it cleanly. Minimum at the origin.
  ravine: {
    f: (x, y) => 0.5 * (x * x + 18 * y * y),
    grad: (x, y) => [x, 18 * y],
    domain: [-4.5, 4.5, -2.2, 2.2],
    min: [0, 0],
    start: [-4, 1.6],
    lr: { sgd: 0.09, momentum: 0.06, rmsprop: 0.12, adam: 0.16 },
  },
};

// ── Optimizers: the exact update rules from lib/*/nanograd, in 2-D ───────────
export type OptName = 'sgd' | 'momentum' | 'rmsprop' | 'adam';

interface Runner {
  name: OptName;
  color: string;
  p: [number, number];
  path: [number, number][];
  // state
  v: [number, number];   // velocity / 1st moment
  s: [number, number];   // 2nd moment (rmsprop/adam)
  t: number;             // step count (adam bias-correction)
  step(surf: SurfaceDef, lrScale: number): void;
}

const OPT_COLORS: Record<OptName, string> = {
  sgd: '#d81b60',       // action red
  momentum: '#f2a900',  // amber
  rmsprop: '#009e73',   // green
  adam: '#6d4aff',      // accent purple
};

function makeRunner(name: OptName, start: [number, number]): Runner {
  return {
    name,
    color: OPT_COLORS[name],
    p: [start[0], start[1]],
    path: [[start[0], start[1]]],
    v: [0, 0],
    s: [0, 0],
    t: 0,
    step(surf, lrScale) {
      const lr = surf.lr[name] * lrScale;
      const [gx, gy] = surf.grad(this.p[0], this.p[1]);
      const g: [number, number] = [gx, gy];
      if (name === 'sgd' || name === 'momentum') {
        const mu = name === 'momentum' ? 0.9 : 0;
        for (let i = 0; i < 2; i++) {
          this.v[i] = mu * this.v[i] + g[i];
          this.p[i] -= lr * this.v[i];
        }
      } else if (name === 'rmsprop') {
        const beta = 0.9, eps = 1e-8;
        for (let i = 0; i < 2; i++) {
          this.s[i] = beta * this.s[i] + (1 - beta) * g[i] * g[i];
          this.p[i] -= (lr * g[i]) / (Math.sqrt(this.s[i]) + eps);
        }
      } else {
        const b1 = 0.9, b2 = 0.999, eps = 1e-8;
        this.t += 1;
        const b1c = 1 - Math.pow(b1, this.t);
        const b2c = 1 - Math.pow(b2, this.t);
        for (let i = 0; i < 2; i++) {
          this.v[i] = b1 * this.v[i] + (1 - b1) * g[i];
          this.s[i] = b2 * this.s[i] + (1 - b2) * g[i] * g[i];
          const mHat = this.v[i] / b1c;
          const vHat = this.s[i] / b2c;
          this.p[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
        }
      }
      // Keep runaway steps from leaving the frame (numerical blowups look ugly).
      this.p[0] = Math.max(-1e4, Math.min(1e4, this.p[0]));
      this.p[1] = Math.max(-1e4, Math.min(1e4, this.p[1]));
      this.path.push([this.p[0], this.p[1]]);
    },
  };
}

// ── Rendering + interaction ──────────────────────────────────────────────────
interface Palette { surface: string; axis: string; accent: string; text: string; }
function readPalette(): Palette {
  const s = getComputedStyle(document.documentElement);
  const v = (n: string) => s.getPropertyValue(n).trim();
  return {
    surface: v('--color-surface') || '#fff',
    axis: v('--viz-axis') || '#bbb',
    accent: v('--viz-accent') || '#6d4aff',
    text: v('--color-muted') || '#666',
  };
}

export interface Stats { closest: OptName; dist: number; step: number; playing: boolean; }
export interface OptHandles {
  reset(): void;
  step(): void;
  toggle(): void;
  setSurface(s: Surface): void;
  setLearningRate(scale: number): void;
  destroy(): void;
}
export interface MountOptions { onStats?: (s: Stats) => void; }

const STEPS_PER_TICK = 3;
const ALL: OptName[] = ['sgd', 'momentum', 'rmsprop', 'adam'];

export function mountOptimizer(canvas: HTMLCanvasElement, opts: MountOptions = {}): OptHandles {
  const maybeCtx = canvas.getContext('2d');
  if (!maybeCtx) throw new Error('2D canvas context unavailable');
  const ctx: CanvasRenderingContext2D = maybeCtx;
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const onStats = opts.onStats ?? (() => {});

  let palette = readPalette();
  let surfaceName: Surface = 'beale';
  let surf = SURFACES[surfaceName];
  let lrScale = 1;
  let start: [number, number] = [...surf.start];
  let runners = ALL.map((n) => makeRunner(n, start));
  let step = 0;
  let playing = false;
  let raf = 0;

  const pad = 8;
  // Offscreen heatmap, recomputed only when the surface/theme/size changes.
  const heat = document.createElement('canvas');

  function toScreen(x: number, y: number): [number, number] {
    const [x0, x1, y0, y1] = surf.domain;
    const sx = ((x - x0) / (x1 - x0)) * (canvas.width - 2 * pad) + pad;
    const sy = canvas.height - (((y - y0) / (y1 - y0)) * (canvas.height - 2 * pad) + pad);
    return [sx, sy];
  }
  function toWorld(sx: number, sy: number): [number, number] {
    const [x0, x1, y0, y1] = surf.domain;
    const x = ((sx - pad) / (canvas.width - 2 * pad)) * (x1 - x0) + x0;
    const y = ((canvas.height - sy - pad) / (canvas.height - 2 * pad)) * (y1 - y0) + y0;
    return [x, y];
  }

  function buildHeat() {
    heat.width = canvas.width;
    heat.height = canvas.height;
    const hctx = heat.getContext('2d');
    if (!hctx) return;
    const img = hctx.createImageData(heat.width, heat.height);
    // Sample log-loss and normalize; blend surface (low) -> accent-tinted (high).
    let lo = Infinity, hi = -Infinity;
    const vals = new Float64Array(heat.width * heat.height);
    for (let sy = 0; sy < heat.height; sy++) {
      for (let sx = 0; sx < heat.width; sx++) {
        const [wx, wy] = toWorld(sx, sy);
        const z = Math.log1p(surf.f(wx, wy));
        vals[sy * heat.width + sx] = z;
        if (z < lo) lo = z;
        if (z > hi) hi = z;
      }
    }
    const [sr, sg, sb] = rgb(palette.surface, [245, 245, 245]);
    const [ar, ag, ab] = rgb(palette.accent, [109, 74, 255]);
    for (let i = 0; i < vals.length; i++) {
      const t = hi > lo ? (vals[i] - lo) / (hi - lo) : 0;
      const m = 0.18 * t; // subtle: high loss tinted toward accent, low = surface
      img.data[i * 4] = sr * (1 - m) + ar * m;
      img.data[i * 4 + 1] = sg * (1 - m) + ag * m;
      img.data[i * 4 + 2] = sb * (1 - m) + ab * m;
      img.data[i * 4 + 3] = 255;
    }
    hctx.putImageData(img, 0, 0);
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(heat, 0, 0);

    // Faint contour rings around the minimum for depth cues.
    const [mx, my] = toScreen(surf.min[0], surf.min[1]);
    ctx.strokeStyle = palette.axis;
    ctx.globalAlpha = 0.35;
    for (let r = 14; r < Math.max(canvas.width, canvas.height); r += 26) {
      ctx.beginPath();
      ctx.arc(mx, my, r, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // The minimum.
    ctx.fillStyle = palette.text;
    ctx.beginPath();
    ctx.arc(mx, my, 4, 0, Math.PI * 2);
    ctx.fill();

    // Each optimizer's trail + head.
    for (const r of runners) {
      ctx.strokeStyle = r.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let k = 0; k < r.path.length; k++) {
        const [sx, sy] = toScreen(r.path[k][0], r.path[k][1]);
        k === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
      }
      ctx.stroke();
      const [hx, hy] = toScreen(r.p[0], r.p[1]);
      ctx.fillStyle = r.color;
      ctx.beginPath();
      ctx.arc(hx, hy, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = palette.surface;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  function dist(r: Runner): number {
    return Math.hypot(r.p[0] - surf.min[0], r.p[1] - surf.min[1]);
  }
  function emit() {
    let best = runners[0];
    for (const r of runners) if (dist(r) < dist(best)) best = r;
    onStats({ closest: best.name, dist: dist(best), step, playing });
  }

  function stepOnce() {
    for (let k = 0; k < STEPS_PER_TICK; k++) {
      for (const r of runners) r.step(surf, lrScale);
    }
    step += STEPS_PER_TICK;
    draw();
    emit();
  }

  let prev = 0, acc = 0;
  const stepMs = 60;
  function loop(now: number) {
    if (!playing) return;
    if (prev === 0) prev = now;
    acc += now - prev;
    prev = now;
    while (acc >= stepMs) { stepOnce(); acc -= stepMs; }
    raf = requestAnimationFrame(loop);
  }
  function play() {
    if (reducedMotion) return;
    playing = true; prev = 0; acc = 0;
    raf = requestAnimationFrame(loop);
    emit();
  }
  function pause() { playing = false; cancelAnimationFrame(raf); emit(); }

  function resetScene() {
    pause();
    runners = ALL.map((n) => makeRunner(n, start));
    step = 0;
    draw();
    emit();
  }

  function onClick(ev: MouseEvent) {
    const rect = canvas.getBoundingClientRect();
    const sx = ((ev.clientX - rect.left) / rect.width) * canvas.width;
    const sy = ((ev.clientY - rect.top) / rect.height) * canvas.height;
    start = toWorld(sx, sy);
    resetScene();
  }
  canvas.addEventListener('click', onClick);

  function onTheme() { palette = readPalette(); buildHeat(); draw(); }
  window.addEventListener('themechange', onTheme);

  buildHeat();
  draw();
  emit();

  return {
    reset: resetScene,
    step() { pause(); stepOnce(); },
    toggle() { playing ? pause() : play(); },
    setSurface(s: Surface) {
      surfaceName = s; surf = SURFACES[s]; start = [...surf.start];
      buildHeat(); resetScene();
    },
    setLearningRate(scale: number) { lrScale = scale; emit(); },
    destroy() {
      pause();
      canvas.removeEventListener('click', onClick);
      window.removeEventListener('themechange', onTheme);
    },
  };
}

// Parse a CSS color (hex or rgb[a]) to [r,g,b], falling back if unreadable.
function rgb(css: string, fallback: [number, number, number]): [number, number, number] {
  if (!css) return fallback;
  const hex = css.match(/^#([0-9a-f]{6})$/i);
  if (hex) {
    const n = parseInt(hex[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  const m = css.match(/rgba?\(([^)]+)\)/i);
  if (m) {
    const parts = m[1].split(',').map((x) => parseFloat(x));
    return [parts[0] || 0, parts[1] || 0, parts[2] || 0];
  }
  return fallback;
}
