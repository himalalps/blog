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
    const parseColumnBoundaries = value => {
      if (typeof value !== 'string') return [];
      const result = new Set();
      for (const part of value.split(',')) {
        const token = part.trim();
        const range = token.match(/^(\d+)\s*-\s*(\d+)$/);
        if (range) {
          const start = Number(range[1]);
          const end = Number(range[2]);
          for (let column = Math.min(start, end); column <= Math.max(start, end); column++) {
            if (column > 0) result.add(column);
          }
        } else if (/^\d+$/.test(token) && Number(token) > 0) {
          result.add(Number(token));
        }
      }
      return [...result];
    };

    const markTableBoundaries = (table, boundaries, mode = 'hide') => {
      if (table.type !== 'table' || boundaries.length === 0) return;
      for (const row of table.children ?? []) {
        for (const [index, cell] of (row.children ?? []).entries()) {
          const column = index + 1;
          const classes = new Set(cell.data?.hProperties?.className ?? []);
          if (mode === 'show') {
            if (column < row.children.length && !boundaries.includes(column)) classes.add('no-vline-right');
            if (column > 1 && !boundaries.includes(column - 1)) classes.add('no-vline-left');
          } else {
            if (boundaries.includes(column)) classes.add('no-vline-right');
            if (boundaries.includes(column - 1)) classes.add('no-vline-left');
          }
          if (classes.size > 0) {
            cell.data = cell.data ?? {};
            cell.data.hProperties = {
              ...(cell.data.hProperties ?? {}),
              className: [...classes]
            };
          }
        }
      }
    };

    const walk = node => {
      for (const child of node.children ?? []) {
        if (child.type === 'containerDirective') {
          child.data = child.data ?? {};
          child.data.hName = 'div';
          const extra =
            typeof child.attributes?.class === 'string'
              ? child.attributes.class.split(/\s+/).filter(Boolean)
              : [];
          if (Object.prototype.hasOwnProperty.call(child.attributes ?? {}, 'fit')) extra.push('fit');
          child.data.hProperties = {
            ...(child.data.hProperties ?? {}),
            className: [child.name, ...extra]
          };
          const hasVlines = Object.prototype.hasOwnProperty.call(child.attributes ?? {}, 'vlines');
          const showBoundaries = parseColumnBoundaries(child.attributes?.vlines);
          const hideBoundaries = parseColumnBoundaries(child.attributes?.['no-vlines']);
          if (hasVlines || hideBoundaries.length > 0) {
            for (const descendant of child.children ?? []) {
              if (descendant.type === 'table') {
                markTableBoundaries(
                  descendant,
                  showBoundaries.length > 0 ? showBoundaries : hideBoundaries,
                  showBoundaries.length > 0 ? 'show' : 'hide'
                );
              }
            }
          }
        }
        walk(child);
      }
    };
    walk(tree);
  };
}
