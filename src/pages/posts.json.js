export async function GET() {
  // Get all posts
  const postModules = import.meta.glob('../content/*.md');

  // Transform posts data
  const postsData = await Promise.all(
    Object.entries(postModules).map(async ([path, resolver]) => {
      const { frontmatter } = await resolver();
      return {
        slug: path.split('/').pop().replace('.md', ''),
        title: frontmatter.title,
        description: frontmatter.description,
        date: frontmatter.date
      };
    })
  );

  return new Response(JSON.stringify(postsData), {
    headers: { 'Content-Type': 'application/json' }
  });
}
