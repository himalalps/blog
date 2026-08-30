/**
 * Turns remark-directive container directives into plain divs, keeping the
 * directive name as a class:
 *
 *   :::booktabs        →  <div class="booktabs"> … </div>
 *   :::hlines my-class →  <div class="hlines my-class"> … </div>
 *
 * Combined with remark-directive (which parses `:::name … :::` blocks),
 * this gives markdown posts a lightweight hook for CSS variants — e.g.
 * table line styles — without dropping down to raw HTML.
 */
export default function remarkDirectiveDivs() {
  return tree => {
    const walk = node => {
      for (const child of node.children ?? []) {
        if (child.type === 'containerDirective') {
          child.data = child.data ?? {};
          child.data.hName = 'div';
          const extra =
            typeof child.attributes?.class === 'string'
              ? child.attributes.class.split(/\s+/).filter(Boolean)
              : [];
          child.data.hProperties = {
            ...(child.data.hProperties ?? {}),
            className: [child.name, ...extra]
          };
        }
        walk(child);
      }
    };
    walk(tree);
  };
}
