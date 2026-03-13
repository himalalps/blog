export async function get() {
  // Get all posts
  const posts = await Astro.glob('../content/posts/*.md');

  // Transform posts data
  const postsData = posts.map(post => ({
    slug: post.file.split('/').pop().replace('.md', ''),
    title: post.frontmatter.title,
    description: post.frontmatter.description,
    date: post.frontmatter.date
  }));

  return {
    body: JSON.stringify(postsData)
  };
}
