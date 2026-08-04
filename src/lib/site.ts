export const basePath = import.meta.env.BASE_URL.replace(/\/$/, '');

export function withBase(path = '') {
  const normalizedPath = path.replace(/^\//, '');
  return normalizedPath ? `${basePath}/${normalizedPath}` : `${basePath}/`;
}

export function getSiteUrl(site: URL) {
  return new URL(withBase(), site);
}
