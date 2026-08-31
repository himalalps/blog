/** Replace a standalone [^giscus] marker with the Giscus comment widget. */
export default function rehypeGiscus() {
  return tree => {
    const giscus = {
      type: 'element', tagName: 'script', properties: { type: 'text/javascript' },
      children: [{ type: 'text', value: `(function() {
  var s = document.createElement('script');
  s.src = 'https://giscus.app/client.js';
  s.dataset.repo = 'himalalps/blog'; s.dataset.repoId = 'R_kgDORl6gRw';
  s.dataset.category = 'Announcements'; s.dataset.categoryId = 'DIC_kwDORl6gR84DEfeQ';
  s.dataset.mapping = 'pathname'; s.dataset.strict = '0'; s.dataset.reactionsEnabled = '1';
  s.dataset.emitMetadata = '0'; s.dataset.inputPosition = 'top';
  s.dataset.theme = document.documentElement.dataset.theme === 'dark' ? 'noborder_dark' : 'noborder_light';
  s.dataset.lang = 'zh-CN'; s.dataset.loading = 'lazy'; s.crossOrigin = 'anonymous'; s.async = true;
  document.currentScript.parentNode.insertBefore(s, document.currentScript.nextSibling);
})();` }],
    };
    const syncTheme = {
      type: 'element', tagName: 'script', properties: { type: 'text/javascript' },
      children: [{ type: 'text', value: `(() => {
  const getTheme = () => document.documentElement.dataset.theme === 'dark' ? 'noborder_dark' : 'noborder_light';
  const update = () => { const frame = document.querySelector('.giscus-frame');
    if (frame) frame.contentWindow.postMessage({ giscus: { setConfig: { theme: getTheme() } } }, 'https://giscus.app'); };
  new MutationObserver(update).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  new MutationObserver(update).observe(document.body, { childList: true, subtree: true }); update();
})();` }],
    };
    const replace = node => {
      if (!node.children) return;
      for (let i = 0; i < node.children.length; i += 1) {
        const child = node.children[i];
        if (child.type === 'element' && child.tagName === 'p'
          && child.children?.length === 1 && child.children[0].type === 'text'
          && child.children[0].value.trim() === '[^giscus]') node.children.splice(i, 1, giscus, syncTheme);
        else replace(child);
      }
    };
    replace(tree);
  };
}
