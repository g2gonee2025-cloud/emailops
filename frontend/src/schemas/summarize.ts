import { z } from 'zod';

// Schema for the data returned by the summarize API endpoint
export const ThreadSummarySchema = z.object({
  summary: z.string().min(1, 'Summary cannot be empty.'),
  key_points: z.array(z.string()).optional(),
  action_items: z.array(z.string()).optional(),
});

// Schema for the form used to request a summary
export const SummarizeFormSchema = z.object({
  threadId: z.string().min(1, 'Thread ID is required.'),
  maxLength: z
    .number()
    .min(100, 'Max length must be at least 100.')
    .max(1000, 'Max length cannot exceed 1000.')
    .default(500),
});

// Inferred TypeScript types
export type ThreadSummary = z.infer<typeof ThreadSummarySchema>;
export type SummarizeFormValues = z.infer<typeof SummarizeFormSchema>;
