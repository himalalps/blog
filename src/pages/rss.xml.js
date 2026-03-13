import rss from '@astrojs/rss';

// In Astro RSS files, we need to use a different approach
export async function GET(context) {
  const posts = import.meta.glob('../content/*.md');

  const postEntries = await Promise.all(
    Object.entries(posts).map(async ([path, resolver]) => {
      const { frontmatter } = await resolver();
      return {
        title: frontmatter.title,
        description: frontmatter.description,
        pubDate: new Date(frontmatter.date),
        link: `${import.meta.env.BASE_URL}/${path.split('/').pop().replace('.md', '')}`,
      };
    })
  );

  return rss({
    title: 'Vibe Blog',
    description: 'A fast, clean, and opinionated technical blog',
    site: context.site,
    items: postEntries,
    customData: `<language>en-us</language>`,
  });
}