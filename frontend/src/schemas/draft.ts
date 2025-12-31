import { z } from 'zod';

/**
 * @description Zod schema for the email draft form.
 * Defines the validation rules for the fields used to generate an email draft.
 */
export const DraftFormSchema = z.object({
  instruction: z.string().min(1, 'Instruction cannot be empty.'),
  threadId: z.string().optional(),
  tone: z.string().optional(),
});

/**
 * @description TypeScript type inferred from the DraftFormSchema.
 * Represents the structure of the data for the email draft form.
 */
export type DraftForm = z.infer<typeof DraftFormSchema>;

/**
 * @description Zod schema for the generated email draft.
 */
export const GeneratedDraftSchema = z.object({
  subject: z.string(),
  body: z.string(),
  to: z.array(z.string()),
  cc: z.array(z.string()),
});

export type GeneratedDraft = z.infer<typeof GeneratedDraftSchema>;
