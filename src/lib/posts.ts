import { getCollection, type CollectionEntry } from 'astro:content';

export type Post = CollectionEntry<'posts'>;

export interface PostSummary {
  slug: string;
  title: string;
  description: string;
  date: string;
  lang: 'zh-CN' | 'en';
}

const validSlug = /^[\p{L}\p{N}]+(?:-[\p{L}\p{N}]+)*$/u;

function comparePostsByDate(left: Post, right: Post) {
  return right.data.date.getTime() - left.data.date.getTime()
    || left.slug.localeCompare(right.slug);
}

export function toPostSummary(post: Post): PostSummary {
  return {
    slug: post.slug,
    title: post.data.title,
    description: post.data.description,
    date: post.data.date.toISOString().slice(0, 10),
    lang: post.data.lang
  };
}

export async function getAllPosts(): Promise<Post[]> {
  const posts = await getCollection('posts');
  for (const post of posts) {
    if (!validSlug.test(post.slug)) {
      throw new Error(`Invalid post slug "${post.slug}". Use letters, numbers, and hyphens between words.`);
    }
  }
  return posts.sort(comparePostsByDate);
}

export async function getPostSummaries(): Promise<PostSummary[]> {
  return (await getAllPosts()).map(toPostSummary);
}
