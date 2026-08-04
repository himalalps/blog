import vue from '@astrojs/vue';
import { defineConfig } from 'astro/config';

// https://astro.build/config
export default defineConfig({
  site: 'https://himalalps.top',
  base: '/blog',
  integrations: [vue()],
  markdown: {
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
