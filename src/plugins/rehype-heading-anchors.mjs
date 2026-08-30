import Slugger from 'github-slugger';

/**
 * Adds GitHub-style `#` anchor links to section headings.
 *
 * Astro's built-in rehypeHeadingIds runs *after* user rehype plugins and only
 * assigns ids to headings that don't have one yet — so we can slug the
 * headings here (with the same github-slugger library) and append a `#`
 * permalink anchor, without breaking Astro's heading collection.
 *
 * The anchor is hidden by default and revealed on heading hover / when the
 * heading is the jump target (:target), styled in global.css.
 */
export default function rehypeHeadingAnchors() {
  return tree => {
    const slugger = new Slugger();

    const textOf = node => {
      if (node.type === 'text') return node.value;
      if (Array.isArray(node.children)) return node.children.map(textOf).join('');
      return '';
    };

    const walk = node => {
      if (!Array.isArray(node.children)) return;
      for (const child of node.children) {
        if (child.type === 'element' && /^h[2-6]$/.test(child.tagName)) {
          if (typeof child.properties?.id !== 'string') {
            const text = textOf(child);
            let slug = slugger.slug(text);
            if (slug.endsWith('-')) slug = slug.slice(0, -1);
            child.properties = { ...child.properties, id: slug };
          }
          child.children.push({
            type: 'element',
            tagName: 'a',
            properties: {
              className: ['heading-anchor'],
              href: `#${child.properties.id}`,
              ariaLabel: '跳转到本节'
            },
            children: [{ type: 'text', value: '#' }]
          });
        }
        walk(child);
      }
    };

    walk(tree);
  };
}
