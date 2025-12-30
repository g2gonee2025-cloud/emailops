import { z } from 'zod';

export const MessageSchema = z.object({
  id: z.string(),
  from: z.string(),
  to: z.array(z.string()),
  subject: z.string(),
  body: z.string(),
  date: z.string(),
});

export type Message = z.infer<typeof MessageSchema>;
