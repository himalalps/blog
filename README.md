# Vibe Technical Blog

A static technical blog built with Astro and Vue. It is published through GitHub Pages at `http://blog.himalalps.top/`.

## Requirements

- Node.js 22
- npm 10 or later

With nvm installed, load it and select the repository version:

```bash
source "$HOME/.nvm/nvm.sh"
nvm use
```

## Development

```bash
npm ci
npm run dev
```

Create a production build with:

```bash
npm run build
```

## Project Structure

```text
src/
├── components/          # Vue search components
├── content/
│   ├── config.ts        # Post frontmatter schema
│   └── posts/
│       └── first-post/
│           └── index.md # Post content and colocated media
├── layouts/             # Shared Astro layout and site-wide scripts
├── lib/                 # Post and site URL helpers
├── pages/               # Astro routes, RSS, and JSON index
├── plugins/             # Markdown citation and heading transforms
└── styles/              # Global styles
```

## Adding a Post

Give every post its own directory. The directory may contain its Markdown, cover, diagrams, screenshots, and other post-specific source files.

```text
src/content/posts/my-new-post/
├── index.md
├── cover.png
└── architecture.png
```

The frontmatter is validated during the build:

```markdown
---
title: "Post Title"
slug: "my-new-post"
date: "2026-08-03"
lang: "zh-CN"
description: "Post description"
cover: "./cover.png"
coverAlt: "Description of the cover image"
---
```

`cover` and `coverAlt` are optional. Reference other colocated images with a relative Markdown path:

```markdown
![Architecture diagram](./architecture.png)
```

Use `lang: "zh-CN"` for Chinese posts and `lang: "en"` for English posts. Slugs may contain Unicode letters, including Chinese characters, as well as numbers and hyphens; Latin slugs remain the most portable choice.

## Math Formulas

Inline and display formulas are rendered with MathJax in the browser. Use dollar delimiters in Markdown:

```markdown
Inline formula: $E = mc^2$.

$$
\int_{-\infty}^{\infty} e^{-x^2} \, dx = \sqrt{\pi}
$$
```

MathJax supports equation labels and same-page references with `\label`, `\ref`, and `\eqref`.

For example, label a display equation and reference it later in the same post:

```markdown
$$
E = mc^2
\label{eq:energy}
$$

As shown in $\eqref{eq:energy}$, mass and energy are equivalent.
```

Several frequently used macros are available globally:

```markdown
$\bm{x} \in \R$, $\E[X]$, $\argmax_x f(x)$, $\argmin_x f(x)$
$\on{softmax}(x)$
```

These expand to `\boldsymbol`, `\mathbb{R}`, `\mathbb{E}`, `\operatorname*{arg\,max}`, `\operatorname*{arg\,min}`, and the built-in `\operatorname{...}` respectively. `\on{...}` is the short alias for `\operatorname{...}`.

Right-clicking a formula opens MathJax's context menu, including **Show MathJax Original Source**. The source dialog supports both horizontal and vertical scrolling. In dark mode, it uses a dark code panel with light text.

## Citations and Bibliography

Add a BibTeX file next to the post and reference it from the frontmatter:

```yaml
bibliography: "ref.bib"
```

Use `[@key]` or `[@key1; @key2]` in the Markdown body. Citations are rendered as linked numeric markers such as `[1]` and `[1,2]`; each marker opens a hover card and links to the corresponding bibliography entry. Citation markers stay attached to the preceding word and are not split across lines.

Add the bibliography placeholder where the references should appear:

```markdown
## References

[^ref]
```

If the placeholder is omitted, the bibliography is appended to the end of the post. Clicking a reference marker scrolls to its entry and highlights it; clicking elsewhere clears that highlight.

MathJax is loaded at runtime from jsDelivr, so formulas require network access to that CDN when a page is opened.

Posts are automatically included in the home page, search, `posts.json`, and RSS feed, sorted newest first. No search index needs to be maintained manually.

## Deployment

Pushes to `main` are built with Node.js 22 and deployed to the `gh-pages` branch by [deploy.yml](.github/workflows/deploy.yml).
