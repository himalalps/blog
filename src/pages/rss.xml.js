import rss from '@astrojs/rss';
import { getPostSummaries } from '../lib/posts';
import { getSiteUrl } from '../lib/site';

export async function GET(context) {
  const posts = await getPostSummaries();
  const siteUrl = getSiteUrl(context.site);

  return rss({
    title: 'Vibe 博客',
    description: '一个快速、简洁且专注内容的技术博客',
    site: siteUrl,
    items: posts.map(post => ({
      title: post.title,
      description: post.description,
      pubDate: new Date(`${post.date}T00:00:00Z`),
      link: new URL(`${post.slug}/`, siteUrl).toString()
    })),
    customData: `<language>zh-cn</language>`,
  });
}
