/** Add a CSS anchor before display math that starts immediately after the
 * preceding block. The anchor survives MathJax's client-side replacement. */
export default function remarkTightMath() {
  return tree => {
    const children = tree.children ?? [];
    for (let index = 0; index < children.length; index += 1) {
      const node = children[index];
      if (node.type !== 'math' || !node.position) continue;
      const previous = children[index - 1];
      if (previous?.position && node.position.start.line === previous.position.end.line + 1) {
        children.splice(index, 0, {
          type: 'html',
          value: '<span class="math-tight-anchor" aria-hidden="true"></span>'
        });
        index += 1;
      }
      const next = children[index + 1];
      if (next?.position && next.position.start.line === node.position.end.line + 1) {
        children.splice(index + 1, 0, {
          type: 'html',
          value: '<span class="math-tight-after-anchor" aria-hidden="true"></span>'
        });
        index += 1;
      }
    }
  };
}
