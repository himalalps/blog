/** Open external links in article content in a new tab. */
export default function rehypeExternalLinks() {
  return tree => {
    const walk = node => {
      if (!node.children) return;
      for (const child of node.children) {
        if (child.type === 'element' && child.tagName === 'a') {
          const href = String(child.properties?.href ?? '');
          if (/^https?:\/\//i.test(href)) {
            child.properties ??= {};
            child.properties.target = '_blank';
            child.properties.rel = 'noopener noreferrer';
          }
        }
        walk(child);
      }
    };
    walk(tree);
  };
}
