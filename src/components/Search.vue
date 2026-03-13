<template>
  <div class="search-container">
    <input
      type="text"
      v-model="searchQuery"
      placeholder="Search blogs..."
      class="search-input"
      @keyup.enter="handleEnter"
    />
    <div v-if="showResults" class="search-results">
      <a
        v-for="post in searchResults"
        :key="post.slug"
        :href="`/${post.slug}`"
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
import { computed, onMounted, ref, watch } from 'vue';

const props = defineProps({
  posts: {
    type: Array,
    default: () => []
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

const handleEnter = () => {
  if (searchQuery.value) {
    window.location.href = `/blog/search?q=${encodeURIComponent(searchQuery.value)}`;
  }
};

// Show results when typing
watch(searchQuery, () => {
  showResults.value = true;
});

// Hide results when clicking outside
onMounted(() => {
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.search-container')) {
      showResults.value = false;
    }
  });
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