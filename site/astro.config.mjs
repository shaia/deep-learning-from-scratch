// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// ─────────────────────────────────────────────────────────────────────────────
// No custom domain yet: the book is served from the GitHub Pages *project page*
// at https://shaia.github.io/deep-learning-from-scratch/. `site` is the origin
// (used for canonical links + sitemap); `base` is the sub-path the repo lives
// under and must match the repo name.
// When a custom subdomain is ready: set SITE_URL to it, change `base` back to
// '/', and re-add `public/CNAME` (+ the DNS record).
// ─────────────────────────────────────────────────────────────────────────────
const SITE_URL = 'https://shaia.github.io';

// https://astro.build/config
export default defineConfig({
  site: SITE_URL,
  // Project page → the site lives under /<repo>/, not the domain root. Internal
  // links/assets use import.meta.env.BASE_URL so they resolve under this base.
  base: '/deep-learning-from-scratch/',
  integrations: [mdx()],
  markdown: {
    // KaTeX: remark-math parses `$…$` / `$$…$$`, rehype-katex renders it to HTML.
    // KaTeX CSS + fonts are imported once in BaseLayout.astro.
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    // Dual-theme syntax highlighting. Astro emits both themes inline and switches
    // via CSS variables; BaseLayout.astro flips them on `html[data-theme]`.
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark' },
      wrap: true,
    },
  },
});
