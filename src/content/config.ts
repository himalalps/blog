import { defineCollection, z } from 'astro:content';

const posts = defineCollection({
  type: 'content',
  schema: ({ image }) => z.object({
    title: z.string().min(1),
    description: z.string().min(1),
    date: z.coerce.date(),
    lang: z.enum(['zh-CN', 'en']).default('zh-CN'),
    cover: image().optional(),
    coverAlt: z.string().min(1).optional()
  })
});

export const collections = { posts };
