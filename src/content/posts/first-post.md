---
title: "Getting Started with Astro"
date: "2026-01-01"
description: "A beginner's guide to Astro framework"
---

# Getting Started with Astro

Astro is a modern static site generator that's fast, flexible, and easy to use. In this post, I'll walk you through the basics of getting started with Astro.

## Why Astro?

Astro offers several key benefits:

- **Fast performance** - Astro generates static HTML by default
- **Component-based architecture** - Use your favorite frontend framework
- **Excellent developer experience** - Hot module replacement and more
- **SEO friendly** - Automatic sitemaps and RSS feeds

## Installation

To install Astro, run the following command:

```bash
npm create astro@latest
```

## Basic Syntax

Here's an example of an Astro component:

```astro
---
const { name } = Astro.props;
---

<div>
  <h1>Hello {name}!</h1>
  <p>Welcome to Astro</p>
</div>
```

## Code Highlighting

Astro supports syntax highlighting out of the box. Here's some JavaScript code:

```javascript
function greet(name) {
  return `Hello, ${name}!`;
}

console.log(greet('World'));
```

And here's some Python:

```python
def greet(name):
    return f"Hello, {name}!"

print(greet("World"))
```

## Conclusion

Astro is a powerful framework for building modern static websites. With its performance benefits and developer-friendly features, it's quickly becoming a popular choice for web developers.

Stay tuned for more posts about Astro and other web technologies!