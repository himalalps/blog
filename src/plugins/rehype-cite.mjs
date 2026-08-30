import path from 'node:path';
import { fileURLToPath } from 'node:url';
import rehypeCitation from 'rehype-citation';

// Vancouver-derived CSL: keeps dots in container-title (arXiv IDs), drops
// "[Internet]" markers and the "Available from:" prefix.
const CSL_PATH = fileURLToPath(new URL('./csl/vancouver.csl', import.meta.url));

/**
 * Per-post rehype citation plugin (wrapper around rehype-citation).
 *
 * Reads the `bibliography` field from the post's frontmatter (path to a .bib
 * file, relative to the markdown file), then renders `[@key]` / `[@a; @b]`
 * citations with:
 *  - vancouver (numeric [n]) CSL style
 *  - inline citations linked to the bibliography entry (linkCitations)
 *  - hover tooltips carrying the full entry (data-tooltip attribute,
 *    rendered as a styled card via CSS)
 *
 * The bibliography is inserted at the `[^ref]` placeholder, or appended
 * at the end of the document if no placeholder is found.
 */
export default function rehypeCite() {
  return async (tree, file) => {
    const fm = file.data?.astro?.frontmatter ?? {};
    const bib = fm.bibliography;
    if (!bib) return;

    const mdPath = file.path || file.history?.[file.history.length - 1];
    if (!mdPath) {
      file.message('[rehype-cite] cannot resolve source file path, skipping');
      return;
    }

    // rehype-citation joins `path` + `bibliography`, so pass the bib filename
    // relative to the markdown file's directory.
    const baseDir = path.isAbsolute(bib) ? path.dirname(bib) : path.dirname(mdPath);
    const bibName = path.isAbsolute(bib) ? path.basename(bib) : bib;
    const result = rehypeCitation({
      bibliography: bibName,
      path: baseDir,
      csl: path.relative(baseDir, CSL_PATH),
      linkCitations: true,
      showTooltips: true,
      tooltipAttribute: 'data-tooltip',
      inlineClass: ['cite-ref'],
    });

    const transform = typeof result === 'function' ? result : result?.[0];
    if (typeof transform === 'function') {
      await transform(tree, file);
      decorateCitations(tree);
    }
  };
}

/**
 * Post-process pass, fixing two rehype-citation shortcomings:
 *
 * 1. `data-tooltip` content is generated per citation via a fresh citeproc
 *    call, so its leading number is always "1." — wrong for [2], [3]…
 * 2. Tooltips fed from an attribute are not real DOM (not selectable).
 *
 * Here we rebuild each hover card from the *final* bibliography section
 * (which carries correct numbering) and attach it as a real element, so the
 * text can be selected/copied and links inside are clickable. The stale
 * data-tooltip attribute is stripped.
 */
export function decorateCitations(tree) {
  const walk = (node, cb) => {
    if (!node.children) return;
    for (const child of node.children) {
      cb(child, node);
      walk(child, cb);
    }
  };

  // 1. Index bibliography entries: key -> { number, body hast nodes }
  const entries = new Map();
  walk(tree, node => {
    const classes = node.properties?.className;
    if (node.type === 'element' && node.tagName === 'div' && Array.isArray(classes) && classes.includes('csl-entry')) {
      const id = String(node.properties?.id ?? '');
      if (!id.startsWith('bib-')) return;
      const left = node.children.find(c => c.properties?.className?.includes?.('csl-left-margin'));
      const right = node.children.find(c => c.properties?.className?.includes?.('csl-right-inline'));
      if (right) {
        const numText = left
          ? left.children.map(c => (typeof c.value === 'string' ? c.value : '')).join('').trim()
          : '';
        const body = linkifyUrls(right.children);
        right.children = body;
        entries.set(id.slice(4), { num: numText, body });
      }
    }
  });
  if (entries.size === 0) return;

  // 2. For each inline citation span: strip data-tooltip, append a real hover card
  walk(tree, (node, parent) => {
    const classes = node.properties?.className;
    if (node.type !== 'element' || node.tagName !== 'span' || !Array.isArray(classes) || !classes.includes('cite-ref')) return;

    // Keep the citation attached to the preceding word, so it cannot become
    // an orphan at the start of the next line. Reuse an existing whitespace
    // character when possible to avoid changing the visible spacing.
    const siblingIndex = parent?.children?.indexOf(node) ?? -1;
    if (siblingIndex > 0) {
      const previous = parent.children[siblingIndex - 1];
      if (previous.type === 'text') {
        if (/\s$/.test(previous.value)) previous.value = previous.value.slice(0, -1);
      }
    }
    node.children.unshift({ type: 'text', value: '\u00a0' });

    const blocks = [];
    const newChildren = [];
    for (const child of node.children) {
      const href = String(child.properties?.href ?? '');
      if (child.type === 'element' && child.tagName === 'a' && href.startsWith('#bib-')) {
        delete child.properties.dataTooltip; // wrong number, no longer needed
        const entry = entries.get(href.slice('#bib-'.length));
        if (entry) {
          blocks.push({
            type: 'element',
            // NOTE: must be a phrasing element (<span>): rehype-raw re-parses the
            // tree with parse5 later, and a <div> inside <p> would be hoisted out.
            tagName: 'span',
            properties: { className: ['cite-pop-entry'] },
            children: [
              ...(entry.num
                ? [{ type: 'element', tagName: 'span', properties: { className: ['cite-pop-num'] }, children: [{ type: 'text', value: `${entry.num} ` }] }]
                : []),
              ...structuredClone(entry.body)
            ]
          });
        }
        newChildren.push(child);
      } else if (child.type === 'text' && !/^[()\s]+$/.test(child.value)) {
        // Separator between multiple citation numbers, e.g. the "," in (1,2).
        // Wrap it so it survives the font-size:0 trick that hides the parens.
        newChildren.push({
          type: 'element',
          tagName: 'span',
          properties: { className: ['cite-sep'] },
          children: [{ type: 'text', value: child.value }]
        });
      } else {
        newChildren.push(child);
      }
    }
    node.children = newChildren;
    if (blocks.length > 0) {
      node.children.push({
        type: 'element',
        tagName: 'span',
        properties: { className: ['cite-pop'] },
        children: blocks
      });
    }
  });
}

/** Wrap bare http(s) URLs inside text nodes with proper <a> elements. */
function linkifyUrls(nodes) {
  const out = [];
  for (const node of nodes) {
    if (node.type !== 'text' || !/https?:\/\//.test(node.value)) {
      out.push(node);
      continue;
    }
    const value = node.value;
    let last = 0;
    for (const m of value.matchAll(/https?:\/\/\S+/g)) {
      if (m.index > last) out.push({ type: 'text', value: value.slice(last, m.index) });
      const raw = m[0];
      const trailing = raw.match(/[.,;:)\]]+$/)?.[0] ?? '';
      const url = raw.slice(0, raw.length - trailing.length);
      out.push({
        type: 'element',
        tagName: 'a',
        properties: { href: url },
        children: [{ type: 'text', value: url }]
      });
      if (trailing) out.push({ type: 'text', value: trailing });
      last = m.index + raw.length;
    }
    if (last < value.length) out.push({ type: 'text', value: value.slice(last) });
  }
  return out;
}
