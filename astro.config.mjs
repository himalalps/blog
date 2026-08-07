import vue from '@astrojs/vue';
import { defineConfig } from 'astro/config';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

// https://astro.build/config
export default defineConfig({
  site: 'http://blog.himalalps.top',
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
