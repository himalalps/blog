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
    title: "Getting Start with Astro",
    description: "A beginner's guide to Astro framework"
  },
  {
    slug: "test-post",
    title: "Getting Started with Astro-2",
    description: "S beginner'side to Astro framework"
  },
  {
    slug: "implementing-search-functionality",
    title: "Implementing Search Functionality in Vibe Blog",
    description: "A step-by-step guide to implementing search functionality in an Astro-based blog"
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

<style scoped>
.search-container {
  position: relative;
  max-width: 400px;
  margin: 0 auto;
}

.search-input {
  width: 100%;
  padding: 0.75rem;
  font-family: 'Courier New', Courier, monospace;
  border: 2px solid var(--border-color);
  background-color: var(--bg-color);
  color: var(--text-color);
  font-size: 1rem;
}

.search-results {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background-color: var(--bg-color);
  border: 2px solid var(--border-color);
  border-top: none;
  z-index: 10;
  max-height: 300px;
  overflow-y: auto;
}

.search-result-item {
  display: block;
  padding: 1rem;
  text-decoration: none;
  color: var(--text-color);
  border-bottom: 1px solid var(--border-color);
}

.search-result-item:last-child {
  border-bottom: none;
}

.search-result-item:hover {
  background-color: var(--code-bg);
}

.search-result-item h3 {
  margin-bottom: 0.5rem;
  font-size: 1rem;
}

.search-result-item p {
  margin: 0;
  font-size: 0.8rem;
  opacity: 0.7;
}

.no-results {
  padding: 1rem;
  text-align: center;
  color: var(--text-color);
  opacity: 0.7;
  border-top: 1px solid var(--border-color);
}
</style>