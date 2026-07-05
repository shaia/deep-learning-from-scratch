// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// ─────────────────────────────────────────────────────────────────────────────
// FILL-IN: set this to the real custom subdomain the book is served from.
// It must be a full, valid URL (used for canonical links + sitemap).
// The same host also lives in `public/CNAME` and your DNS CNAME record.
// ─────────────────────────────────────────────────────────────────────────────
const SITE_URL = 'https://book.example.com';

// https://astro.build/config
export default defineConfig({
  site: SITE_URL,
  // Custom subdomain → the book is served from the domain root, so base is '/'.
  // (If you ever fall back to a project page at user.github.io/<repo>/,
  //  change this to '/<repo>/' and rebuild.)
  base: '/',
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
