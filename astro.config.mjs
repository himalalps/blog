import vue from '@astrojs/vue';
import { defineConfig } from 'astro/config';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

// https://astro.build/config
export default defineConfig({
  site: 'https://himalalps.top',
  base: '/blog',
  integrations: [vue()],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    syntaxHighlight: 'shiki',
    shikiConfig: {
      theme: 'nord',
      wrap: true
    }
  },
  devToolbar: {
    enabled: false
  }
});
