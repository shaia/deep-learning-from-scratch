// Reference widget logic — vanilla TypeScript + Canvas, no framework.
// It is deliberately content-free (not tied to any module); it exists to lock
// in the pattern every real widget will follow:
//   • read semantic colors from the site theme (CSS variables), never hardcode
//   • repaint on `themechange` (dispatched by BaseLayout's theme toggle)
//   • respect prefers-reduced-motion
//   • deterministic seed + a reset control
//
// A real module widget replaces `draw()` / `step()` with its own mechanism and
// wires its controls to the same lifecycle.

/** Semantic colors pulled live from the current theme (see viz-style.md). */
interface Palette {
  bg: string;
  axis: string;
  dataA: string;
  dataB: string;
  accent: string;
}

function readPalette(): Palette {
  const s = getComputedStyle(document.documentElement);
  const v = (name: string) => s.getPropertyValue(name).trim();
  return {
    bg: v('--color-surface'),
    axis: v('--viz-axis'),
    dataA: v('--viz-data-a'),
    dataB: v('--viz-data-b'),
    accent: v('--viz-accent'),
  };
}

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

interface Dot {
  x: number;
  y: number;
  vx: number;
  cls: 0 | 1;
}

const SEED = 1234;

export interface WidgetHandles {
  reset: () => void;
  destroy: () => void;
}

export function mountExampleWidget(canvas: HTMLCanvasElement): WidgetHandles {
  const maybeCtx = canvas.getContext('2d');
  if (!maybeCtx) throw new Error('2D canvas context unavailable');
  // Bind with an explicit non-null type so the guard survives into the nested
  // draw/step closures (TS won't narrow a captured variable on its own).
  const ctx: CanvasRenderingContext2D = maybeCtx;

  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let dots: Dot[] = [];
  let palette = readPalette();
  let raf = 0;

  function seedDots() {
    const rand = mulberry32(SEED);
    dots = Array.from({ length: 24 }, () => ({
      x: rand(),
      y: 0.15 + rand() * 0.7,
      vx: 0.0015 + rand() * 0.0025,
      cls: rand() > 0.5 ? 1 : 0,
    }));
  }

  function draw() {
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = palette.bg;
    ctx.fillRect(0, 0, width, height);

    // baseline axis
    ctx.strokeStyle = palette.axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    for (const d of dots) {
      ctx.beginPath();
      ctx.fillStyle = d.cls === 0 ? palette.dataA : palette.dataB;
      ctx.arc(d.x * width, d.y * height, 5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function step() {
    for (const d of dots) {
      d.x += d.vx;
      if (d.x > 1) d.x = 0;
    }
    draw();
    raf = requestAnimationFrame(step);
  }

  function start() {
    cancelAnimationFrame(raf);
    if (reducedMotion) {
      draw(); // static frame — no motion
    } else {
      raf = requestAnimationFrame(step);
    }
  }

  function onTheme() {
    palette = readPalette();
    draw();
  }
  window.addEventListener('themechange', onTheme);

  seedDots();
  start();

  return {
    reset() {
      seedDots();
      start();
    },
    destroy() {
      cancelAnimationFrame(raf);
      window.removeEventListener('themechange', onTheme);
    },
  };
}
