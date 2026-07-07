# Publishing: the Astro book + the slim Ghost companion

The book has two front doors, with a clear division of labor:

- **Astro on GitHub Pages** (`book.<yourdomain>`) — the **canonical home** of every module.
  The full thing: MDX prose, KaTeX math, side-by-side C/Python, interactive widgets, Manim
  embeds. It is versioned in this repo (`site/`) and is the source of truth.
- **Ghost Pro** (managed) — a **slim companion**. It reuses the existing audience and
  newsletter. Each module gets a short teaser post there that links out to the full Astro
  module. Ghost is *not* where the module lives — it's how readers discover it and how it
  reaches subscribers' inboxes.

Ghost(Pro) is managed hosting with no server access, so the Astro site cannot be deployed
onto it. The two stay separate and cross-link. That separation is deliberate: Ghost's editor
can't hold MDX components, KaTeX, or live widgets, so the deep content belongs on Astro.

## Per-module flow

1. **Write & ship the full module in Astro.** Author `site/src/content/posts/NN-slug.mdx`
   (structure per [`new-module.md`](new-module.md) Step 6), set `draft: false`, push to
   `main`. CI deploys it to `https://book.<yourdomain>/posts/NN-slug/`.
2. **Write a slim Ghost teaser.** Hook + one or two paragraphs of intuition — the "why did
   anyone need this?" framing, no math or code. End with a call to action linking to the
   Astro module: *"Read the full interactive module → book.<yourdomain>/posts/NN-slug/"*.
3. **Set the Ghost post's canonical URL to the Astro URL.** In Ghost: post settings →
   *Meta data* / *Canonical URL* → paste the Astro module URL. This is the one real gotcha of
   dual-publishing — it tells search engines the Astro page is authoritative so the teaser
   doesn't compete with it or split ranking signals.
4. **Record the back-link in the Astro post.** Set `ghostUrl` in the MDX frontmatter to the
   Ghost teaser URL; `PostLayout` then shows a "read the short version & subscribe" link.
5. **Send it as the newsletter.** Publish the Ghost post to the email list to reach existing
   subscribers.

## Why this split

- Keeps the interactive, math-heavy content on the platform that can render it (Astro).
- Reuses the Ghost audience/newsletter without duplicating or degrading the real content.
- Canonical-to-Astro means the git-versioned page is always the authoritative one.

## Domain notes

- Astro is served from a **custom subdomain** (`book.<yourdomain>`) via GitHub Pages: the host
  is set in `site/astro.config.mjs` (`site`) and `site/public/CNAME`, with a DNS **CNAME**
  record `book → <username>.github.io`. Ghost keeps the apex/primary domain.
- If you ever drop the custom domain, switch to the project-page URL
  (`<username>.github.io/<repo>/`) by setting `base: '/<repo>/'` in `astro.config.mjs` and
  removing `public/CNAME`.
