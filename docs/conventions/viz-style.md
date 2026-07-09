# Visualization Style Guide

Two animation media, one visual language. Whether a figure is an interactive web widget or a
Manim video, it should feel like it belongs to the same book.

## The two media

- **Interactive widgets** (vanilla TS + Canvas/SVG, embedded as Astro islands): for things the
  reader should *manipulate* — drag data points, move a slider, step training, toggle a feature.
- **Manim videos**: for *narrated, linear* explanations — a derivation unfolding, a mechanism
  animating once, cinematic concept intros.

Rule of thumb: if the insight comes from *poking at it*, make a widget. If it comes from
*watching it explained in order*, make a Manim scene. Most modules want one of each.

## Shared visual language

- **Theme-aware.** Both media must read in light and dark. Widgets respond to the site theme;
  Manim scenes use a neutral background that embeds cleanly either way.
- **Semantic color roles** (fill in exact hex once the site theme lands; keep them consistent):
  - *neutral / axes / text* — low-contrast gray
  - *data / points* — one categorical pair for two classes (colorblind-safe)
  - *model / prediction* — the "accent" color
  - *gradient / update* — a single "action" color used only for change/motion
  - *positive vs negative* — a diverging pair (e.g. for weights, gradients)
- **Colorblind-safe** categorical pairs; never encode by color alone — add shape/label.
- **Motion language:** ease-in-out, ~300–600 ms transitions; motion should *mean* something
  (a value changed), never decoration. Show the update, then settle.

## Widget conventions

- One idea per widget; a short caption stating what to try.
- Controls labelled with the same symbols as the math (`lr`, `w`, …).
- Deterministic seed + a "reset" control. Runs at 60 fps on a laptop; degrade gracefully.
- Keyboard-accessible controls; `prefers-reduced-motion` respected.

## Manim conventions

- 16:9, medium quality for drafts (`-qm`), high for final (`-qh`).
- LaTeX labels match `math-notation.md`.
- Rendered assets flow through `animations/` → the site's public assets; embed as MP4 with a
  poster image and captions.
- Keep scenes short (30–90 s) and single-purpose; prefer several small scenes over one long one.

## Accessibility & fallback

- Every animation has a one-line text description of what it shows.
- If Manim/LaTeX tooling is unavailable, a matplotlib-generated GIF is an acceptable fallback
  for simple scenes (noted per module).
