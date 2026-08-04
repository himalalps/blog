# Vibe Technical Blog

A static technical blog built with Astro and Vue. It is published through GitHub Pages at `https://himalalps.top/blog/`.

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
├── layouts/             # Shared Astro layout
├── lib/                 # Post and site URL helpers
├── pages/               # Astro routes, RSS, and JSON index
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

Posts are automatically included in the home page, search, `posts.json`, and RSS feed, sorted newest first. No search index needs to be maintained manually.

## Deployment

Pushes to `main` are built with Node.js 22 and deployed to the `gh-pages` branch by [deploy.yml](.github/workflows/deploy.yml).
