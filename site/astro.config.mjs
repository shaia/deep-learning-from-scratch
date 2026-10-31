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
const BASE = '/deep-learning-from-scratch/';

/**
 * Posts link to each other with plain markdown, root-relative —
 * `[LSTM →](/posts/07-lstm/)` — as if the book lived at the domain root. This
 * rehype plugin resolves those hrefs under `base`, as Astro does for its own
 * routes, so they work on the project page and keep working if `base` returns
 * to '/'. Only markdown links are touched: protocol-relative (`//…`) and
 * already-based hrefs are left alone, and JSX in MDX keeps using
 * import.meta.env.BASE_URL.
 * @param {string} base
 */
function rehypeBaseLinks(base) {
  const prefix = base.replace(/\/$/, '');
  /** @param {any} node */
  const visit = (node) => {
    const href = node.type === 'element' && node.tagName === 'a' ? node.properties?.href : null;
    if (typeof href === 'string' && href.startsWith('/') && !href.startsWith('//')
        && !href.startsWith(base)) {
      node.properties.href = prefix + href;
    }
    for (const child of node.children ?? []) visit(child);
  };
  return () => visit;
}

// https://astro.build/config
export default defineConfig({
  site: SITE_URL,
  // Project page → the site lives under /<repo>/, not the domain root. Internal
  // links/assets use import.meta.env.BASE_URL so they resolve under this base;
  // markdown links in posts are resolved by rehypeBaseLinks above.
  base: BASE,
  integrations: [mdx()],
  markdown: {
    // KaTeX: remark-math parses `$…$` / `$$…$$`, rehype-katex renders it to HTML.
    // KaTeX CSS + fonts are imported once in BaseLayout.astro.
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex, rehypeBaseLinks(BASE)],
    // Dual-theme syntax highlighting. Astro emits both themes inline and switches
    // via CSS variables; BaseLayout.astro flips them on `html[data-theme]`.
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark' },
      wrap: true,
    },
  },
});
