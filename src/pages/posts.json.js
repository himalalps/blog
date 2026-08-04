import { getPostSummaries } from '../lib/posts';

export async function GET() {
  const posts = await getPostSummaries();

  return new Response(JSON.stringify(posts), {
    headers: { 'Content-Type': 'application/json; charset=utf-8' }
  });
}
