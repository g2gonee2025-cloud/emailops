import { z } from 'zod';

export const DraftSchema = z.object({
  to: z.array(z.string().email()).min(1, 'At least one recipient is required'),
  cc: z.array(z.string().email()).optional(),
  bcc: z.array(z.string().email()).optional(),
  subject: z.string().min(1, 'Subject is required'),
  body: z.string().min(1, 'Body is required'),
});

export type Draft = z.infer<typeof DraftSchema>;

export const isInvalidEmail = (email: string) => {
    return !z.string().email().safeParse(email).success;
}

export const DraftGenerationFormSchema = z.object({
    instruction: z.string().min(1, 'Instruction is required'),
    threadId: z.string().optional(),
});

export type DraftGenerationForm = z.infer<typeof DraftGenerationFormSchema>;
