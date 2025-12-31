import { z } from 'zod';

// ================================================================================
// Zod Schemas
// ================================================================================

/**
 * Schema for a single source of information, as provided by the backend API.
 */
export const SourceSchema = z.object({
  chunk_id: z.string().uuid('Invalid chunk ID'),
  content: z.string(),
  score: z.number().min(0).max(1, 'Score must be between 0 and 1'),
});

/**
 * Schema for a single message in the AskView chat interface.
 */
export const MessageSchema = z.object({
  role: z.enum(['user', 'assistant']),
  content: z.string(),
  sources: z.array(SourceSchema).optional(),
  timestamp: z.date(),
});

/**
 * Schema for the chat history, which is an array of messages.
 */
export const ChatHistorySchema = z.array(MessageSchema);

/**
 * Schema for the chat message format expected by the backend API.
 */
export const ApiChatMessageSchema = z.object({
  role: z.enum(['system', 'user', 'assistant']),
  content: z.string(),
});

/**
 * Schema for the form used to submit a new question in the AskView.
 */
export const AskFormSchema = z.object({
  input: z.string().trim().min(1, 'Message cannot be empty'),
});

// ================================================================================
// Inferred TypeScript Types
// ================================================================================

export type Source = z.infer<typeof SourceSchema>;
export type Message = z.infer<typeof MessageSchema>;
export type ChatHistory = z.infer<typeof ChatHistorySchema>;
export type ApiChatMessage = z.infer<typeof ApiChatMessageSchema>;
export type AskForm = z.infer<typeof AskFormSchema>;

// ================================================================================
// Validation Helpers
// ================================================================================

export const validateMessage = (data: unknown): Message | null => {
  const result = MessageSchema.safeParse(data);
  return result.success ? result.data : null;
};

export const validateAskForm = (data: unknown): AskForm | null => {
  const result = AskFormSchema.safeParse(data);
  return result.success ? result.data : null;
};
