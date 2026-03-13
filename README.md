# Vibe - Technical Blog

A fast, clean, and opinionated technical blog built with Astro and Vue.js.

## Features

- **Fast** - Built with Astro's static site generation for optimal performance
- **Clean** - Minimalist design with a focus on content
- **Search** - Client-side search functionality with real-time preview
- **Responsive** - Works on all device sizes
- **Syntax Highlighting** - Code blocks with syntax highlighting and copy functionality

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd vibe
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Run the development server**
   ```bash
   npm run dev
   ```

4. **Build for production**
   ```bash
   npm run build
   ```

## Project Structure

```
vibe/
├── src/
│   ├── components/       # Vue components
│   │   └── Search.vue    # Search component with real-time preview
│   ├── content/          # Markdown content
│   │   └── posts/        # Blog posts
│   ├── layouts/          # Astro layouts
│   │   └── Layout.astro  # Main layout component
│   ├── pages/            # Astro pages
│   │   ├── index.astro   # Home page
│   │   ├── posts/        # Posts page and dynamic routes
│   │   ├── search/       # Search results page
│   │   └── rss.xml.js    # RSS feed
│   └── styles/           # CSS styles
│       └── global.css    # Global styles
├── astro.config.mjs      # Astro configuration
├── package.json          # Package configuration
└── README.md             # This file
```

## Search Functionality

### How it works

1. **Real-time preview** - When typing in the search box, a dropdown appears with matching posts
2. **Search results page** - When pressing Enter, you're taken to a dedicated search results page
3. **Case-insensitive** - Search is case-insensitive for better results

### Adding New Posts

To add a new post:

1. **Create a new Markdown file** in `src/content/posts/`
   - Example: `new-post.md`

2. **Add frontmatter** to the file:
   ```markdown
   ---
   title: "Post Title"
   date: "2026-03-12"
   description: "Post description"
   ---
   
   # Post Content
   ```

3. **Update the search data** in `src/components/Search.vue`:
   ```javascript
   // Hardcoded posts data
   const posts = [
     {
       slug: "first-post",
       title: "Getting Started with Astro",
       description: "A beginner's guide to Astro framework"
     },
     {
       slug: "test-post",
       title: "Getting Started with Astro-2",
       description: "S beginner'side to Astro framework"
     },
     {
       slug: "new-post",  // Add new post here
       title: "Post Title",
       description: "Post description"
     }
   ];
   ```

4. **Rebuild the project**
   ```bash
   npm run build
   ```

## Technologies Used

- **Astro** - Static site generator
- **Vue.js** - Frontend framework for interactive components
- **Markdown** - Content format
- **CSS** - Styling

## Customization

### Changing the theme

Update the CSS variables in `src/styles/global.css` to customize colors and fonts.

### Adding new components

Create new Vue components in `src/components/` and import them in your Astro pages.

## License

MIT
