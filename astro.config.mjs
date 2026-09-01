import vue from '@astrojs/vue';
import { defineConfig } from 'astro/config';
import rehypeMathjaxBrowser from 'rehype-mathjax/browser';
import remarkMath from 'remark-math';
import remarkDirective from 'remark-directive';
import remarkDirectiveDivs from './src/plugins/remark-directive-divs.mjs';
import remarkTightMath from './src/plugins/remark-tight-math.mjs';
import rehypeCite from './src/plugins/rehype-cite.mjs';
import rehypeHeadingAnchors from './src/plugins/rehype-heading-anchors.mjs';
import rehypeExternalLinks from './src/plugins/rehype-external-links.mjs';
import rehypeGiscus from './src/plugins/rehype-giscus.mjs';

// https://astro.build/config
export default defineConfig({
  site: 'http://blog.himalalps.top',
  integrations: [vue()],
  markdown: {
    remarkPlugins: [remarkMath, remarkDirective, remarkDirectiveDivs, remarkTightMath],
    rehypePlugins: [rehypeMathjaxBrowser, rehypeCite, rehypeGiscus, rehypeExternalLinks, rehypeHeadingAnchors],
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
