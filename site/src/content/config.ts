import { defineCollection, z } from 'astro:content';

// One entry per curriculum module. Files live at
// src/content/posts/NN-slug.mdx and follow the intuition → history → math →
// code → play structure from docs/conventions/new-module.md (Step 6).
const posts = defineCollection({
  type: 'content',
  schema: z.object({
    // Display title, e.g. "The Perceptron".
    title: z.string(),
    // Zero-padded curriculum number as a NUMBER for ordering (0, 1, 2, …, 16).
    moduleNumber: z.number().int().min(0),
    // NOTE: the URL slug is NOT declared here — Astro reserves `slug` and derives
    // it from the filename. Name files "NN-slug.mdx" (e.g. 00-perceptron.mdx) and
    // the entry's `slug` becomes "NN-slug" automatically.
    // One-line summary — used for the TOC card and <meta name="description">.
    description: z.string(),
    // Hide from the index / nav while a module is still in progress.
    draft: z.boolean().default(true),
    // OPTIONAL back-link to the slim Ghost teaser for this module.
    // When set, PostLayout renders a "read/subscribe on the newsletter" link.
    // See docs/conventions/publishing.md for the Ghost ↔ Astro relationship.
    ghostUrl: z.string().url().optional(),
  }),
});

export const collections = { posts };
