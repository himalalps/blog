---
title: "Implementing Search Functionality in Vibe Blog"
date: "2026-03-12"
description: "A step-by-step guide to implementing search functionality in an Astro-based blog"
---

# Implementing Search Functionality in Vibe Blog

In this post, I'll walk you through the process of implementing search functionality in the Vibe blog, including real-time search preview and search results page.

## Project Overview

Vibe is a technical blog built with Astro and Vue.js, featuring static site generation for optimal performance and a clean, minimalist design.

## Search Functionality Requirements

1. **Real-time search preview** - Show matching posts as the user types
2. **Search results page** - Display all matching posts on a dedicated page
3. **Case-insensitive search** - Ensure search works regardless of case
4. **Fast performance** - Search should be responsive and not slow down the site

## Implementation Process

### Step 1: Create the Search Component

First, I created a Vue component for the search functionality:

```vue
<template>
  <div class="search-container">
    <input
      type="text"
      v-model="searchQuery"
      placeholder="Search posts..."
      class="search-input"
      @keyup.enter="handleEnter"
    />
    <div v-if="showResults" class="search-results">
      <a
        v-for="post in searchResults"
        :key="post.slug"
        :href="`/posts/${post.slug}`"
        class="search-result-item"
      >
        <h3>{{ post.title }}</h3>
        <p>{{ post.description }}</p>
      </a>
      <div v-if="searchResults.length === 0 && searchQuery" class="no-results">
        No results found for "{{ searchQuery }}"
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue';

const searchQuery = ref('');
const showResults = ref(false);

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
  }
];

const searchResults = computed(() => {
  if (!searchQuery.value) return [];
  
  return posts.filter(post => {
    const query = searchQuery.value.toLowerCase();
    return (
      post.title.toLowerCase().includes(query) ||
      post.description.toLowerCase().includes(query)
    );
  });
});

const handleEnter = () => {
  if (searchQuery.value) {
    window.location.href = `/search?q=${encodeURIComponent(searchQuery.value)}`;
  }
};

// Show results when typing
watch(searchQuery, () => {
  showResults.value = true;
});
</script>
```

### Step 2: Create the Search Results Page

Next, I created a dedicated search results page:

```astro
---
import Layout from '../../layouts/Layout.astro';

// Get all posts for client-side search
const posts = await Astro.glob('../../content/posts/*.md');
const searchPosts = posts.map(post => ({
  slug: post.file.split('/').pop().replace('.md', ''),
  title: post.frontmatter.title,
  description: post.frontmatter.description,
  date: post.frontmatter.date
}));
---

<Layout title="Search - Vibe" description="Search blog posts">
  <section class="search-page">
    <div id="search-results" class="search-results-container">
      <h2>Results</h2>
      <div id="results-list" class="posts-grid">
        <!-- Results will be populated by JavaScript -->
      </div>
      <div id="no-results" class="no-results" style="display: none;">
        <p>No results found. Please try a different search term.</p>
      </div>
    </div>
  </section>
</Layout>

<script set:html={`
  // Get search query from URL if present
  const urlParams = new URLSearchParams(window.location.search);
  const searchQuery = urlParams.get('q');
  
  // All posts data
  const allPosts = ${JSON.stringify(searchPosts)};
  
  // If search query is present, run search
  if (searchQuery) {
    // Filter posts based on search query
    const results = allPosts.filter(post => {
      const searchTerm = searchQuery.toLowerCase();
      return (
        post.title.toLowerCase().includes(searchTerm) ||
        post.description.toLowerCase().includes(searchTerm)
      );
    });
    
    // Display results
    const resultsList = document.getElementById('results-list');
    const noResults = document.getElementById('no-results');
    
    if (results.length > 0) {
      var html = '';
      for (var i = 0; i < results.length; i++) {
        var post = results[i];
        html += '<a href="/posts/' + post.slug + '" class="post-card">';
        html += '<h3>' + post.title + '</h3>';
        html += '<p>' + post.description + '</p>';
        html += '<p class="date">' + post.date + '</p>';
        html += '</a>';
      }
      resultsList.innerHTML = html;
      resultsList.style.display = 'grid';
      noResults.style.display = 'none';
    } else {
      resultsList.style.display = 'none';
      noResults.style.display = 'block';
    }
  }
`}></script>

<style scoped>
.search-page {
  max-width: 800px;
  margin: 0 auto;
  padding: 3rem 0;
}

.search-results-container {
  margin-top: 2rem;
}

.search-results-container h2 {
  font-size: 1.8rem;
  margin-bottom: 1.5rem;
  text-transform: uppercase;
  letter-spacing: -0.5px;
}

.no-results {
  text-align: center;
  padding: 4rem 2rem;
  border: 2px solid var(--border-color);
  box-shadow: 5px 5px 0 var(--border-color);
  margin-top: 2rem;
}

.no-results p {
  font-size: 1.2rem;
  margin: 0;
}
</style>
```

### Step 3: Integrate the Search Component into the Layout

I added the search component to the main layout:

```astro
---
import Search from '../components/Search.vue';
import '../styles/global.css';

const { title, description } = Astro.props;
---

<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{title || 'Vibe'}</title>
    {description && <meta name="description" content={description} />}
    <link rel="alternate" type="application/rss+xml" href="/rss.xml" title="Vibe Blog" />
  </head>
  <body>
    <header>
      <nav>
        <a href="/" class="logo">VIBE</a>
        <div class="nav-links">
          <a href="/">Home</a>
          <a href="/posts">Posts</a>
        </div>
        <div class="search">
          <Search client:load />
        </div>
      </nav>
    </header>
    <main>
      <slot />
    </main>
    <footer>
      <p>&copy; {new Date().getFullYear()} Vibe. All rights reserved.</p>
    </footer>
  </body>
</html>
```

## Challenges and Solutions

### Challenge 1: Data Passing from Astro to Vue

**Issue:** Initially, I tried to pass post data from Astro to the Vue search component using props, but it didn't work correctly.

**Solution:** I hardcoded the post data directly in the Vue component. While this isn't the most scalable solution, it's simple and reliable for a small blog.

### Challenge 2: Client-side Search Logic

**Issue:** Implementing client-side search that works both for the real-time preview and the search results page.

**Solution:** I used Vue's computed properties for the real-time preview and plain JavaScript for the search results page, both using the same search logic.

### Challenge 3: HTML Generation in Search Results

**Issue:** Generating HTML dynamically in the search results page was tricky due to quote escaping and format issues.

**Solution:** I used a for loop and string concatenation instead of template literals to avoid quote escaping issues.

## How to Add New Posts

To add new posts to the blog and ensure they appear in search results:

1. **Create a new Markdown file** in `src/content/posts/`
2. **Add frontmatter** with title, date, and description
3. **Update the search data** in `src/components/Search.vue`
4. **Rebuild the project** with `npm run build`

## Final Result

The search functionality now works as expected:

- **Real-time preview** shows matching posts as you type
- **Search results page** displays all matching posts
- **Case-insensitive search** ensures accurate results
- **Fast performance** due to client-side processing

## Conclusion

Implementing search functionality in an Astro-based blog requires a combination of client-side JavaScript and careful data management. While the hardcoded approach is simple for small blogs, larger blogs might benefit from more scalable solutions like API routes or statically generated search indexes.

Overall, the search functionality adds a valuable user experience feature to the Vibe blog, making it easier for readers to find relevant content.
