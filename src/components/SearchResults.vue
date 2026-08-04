<template>
  <section class="search-page">
    <h1>搜索结果</h1>
    <p v-if="searchQuery" class="search-query">“{{ searchQuery }}”的搜索结果</p>

    <div v-if="searchResults.length" class="posts-grid">
      <a
        v-for="post in searchResults"
        :key="post.slug"
        :href="postHref(post.slug)"
        class="post-card"
      >
        <h2>{{ post.title }}</h2>
        <p>{{ post.description }}</p>
        <p class="date">{{ post.date }}</p>
      </a>
    </div>

    <div v-else-if="searchQuery" class="no-results">
      <p>没有找到与“{{ searchQuery }}”相关的文章。</p>
    </div>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue';

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

const searchResults = computed(() => {
  const query = searchQuery.value.toLowerCase();
  if (!query) return [];

  return props.posts.filter((post) => (
    post.title.toLowerCase().includes(query)
    || post.description.toLowerCase().includes(query)
  ));
});

const postHref = (slug) => `${props.baseUrl}/${slug}`;

onMounted(() => {
  searchQuery.value = new URLSearchParams(window.location.search).get('q')?.trim() ?? '';
});
</script>

<style scoped>
.search-page {
  max-width: 800px;
  margin: 0 auto;
  padding: 3rem 0;
}

.search-page h1 {
  font-size: 1.8rem;
  text-transform: uppercase;
}

.search-query {
  margin-top: 0.75rem;
  opacity: 0.7;
}

.post-card h2 {
  margin-bottom: 1rem;
  font-size: 1.4rem;
  line-height: 1.3;
  text-transform: uppercase;
}

.no-results {
  margin-top: 2rem;
  padding: 4rem 2rem;
  text-align: center;
  border: 2px solid var(--border-color);
  box-shadow: 5px 5px 0 var(--border-color);
}
</style>
