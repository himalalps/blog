<template>
  <div class="search-container">
    <input
      type="search"
      v-model="searchQuery"
      placeholder="搜索文章..."
      aria-label="搜索博客文章"
      autocomplete="off"
      class="search-input"
      @focus="showResults = Boolean(searchQuery)"
      @keydown.esc="showResults = false"
      @keyup.enter="handleEnter"
    />
    <div v-if="showResults && searchQuery" class="search-results">
      <a
        v-for="post in searchResults"
        :key="post.slug"
        :href="postHref(post.slug)"
        class="search-result-item"
      >
        <h3>{{ post.title }}</h3>
        <p>{{ post.description }}</p>
      </a>
      <div v-if="searchResults.length === 0 && searchQuery" class="no-results">
        没有找到与“{{ searchQuery }}”相关的文章
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';

const props = defineProps({
  posts: {
    type: Array,
    default: () => []
  },
  baseUrl: {
    type: String,
    default: ''
  }
});

const searchQuery = ref('');
const showResults = ref(false);

const searchResults = computed(() => {
  if (!searchQuery.value) return [];
  
  return props.posts.filter(post => {
    const query = searchQuery.value.toLowerCase();
    return (
      post.title.toLowerCase().includes(query) ||
      post.description.toLowerCase().includes(query)
    );
  });
});

const postHref = (slug) => `${props.baseUrl}/${slug}`;

const handleEnter = () => {
  const query = searchQuery.value.trim();
  if (query) {
    window.location.href = `${props.baseUrl}/search?q=${encodeURIComponent(query)}`;
  }
};

// Show results when typing
watch(searchQuery, () => {
  showResults.value = true;
});

const handleDocumentClick = (event) => {
  if (!event.target.closest('.search-container')) {
    showResults.value = false;
  }
};

onMounted(() => document.addEventListener('click', handleDocumentClick));
onBeforeUnmount(() => document.removeEventListener('click', handleDocumentClick));
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
  font-family: var(--font-body);
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
