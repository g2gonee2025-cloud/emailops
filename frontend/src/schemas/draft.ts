/**
 * Zod schemas for the DraftView.
 *
 * This file defines the validation schemas for the draft email form and the
 * expected API response structure. It also exports the inferred TypeScript types
 * for type safety throughout the application.
 */
import { z } from 'zod';

// =============================================================================
// Core Schemas
// =============================================================================

/**
 * @description Defines the allowed tones for an email draft.
 */
export const ToneSchema = z.enum([
  'professional',
  'friendly',
  'formal',
  'concise',
]);

/**
 * @description Schema for the form used to request a new email draft.
 * - `instruction`: The user's prompt for what the email should contain. Must be a non-empty string.
 * - `threadId`: An optional ID to provide context from an existing email thread.
 * - `tone`: The desired tone of the email, conforming to the ToneSchema.
 */
export const DraftFormSchema = z.object({
  instruction: z.string().min(1, 'Instruction cannot be empty.'),
  threadId: z.string().optional(),
  tone: ToneSchema.default('professional'),
});

/**
 * @description Schema for the `EmailDraft` object returned by the API.
 * - `subject`: The generated subject line of the email.
 * - `body`: The generated body content of the email.
 * - `to`: An optional array of recipient email addresses.
 * - `cc`: An optional array of CC'd recipient email addresses.
 */
export const EmailDraftSchema = z.object({
  subject: z.string(),
  body: z.string(),
  to: z.array(z.string()).optional(),
  cc: z.array(z.string()).optional(),
});

/**
 * @description Schema for the full API response when drafting an email.
 * - `correlation_id`: An optional ID for tracing the request.
 * - `draft`: The generated `EmailDraft` object.
 * - `confidence`: A numerical score indicating the AI's confidence in the draft.
 * - `iterations`: The number of iterations the AI took to generate the draft.
 */
export const DraftEmailResponseSchema = z.object({
  correlation_id: z.string().optional(),
  draft: EmailDraftSchema,
  confidence: z.number(),
  iterations: z.number(),
});


// =============================================================================
// Inferred TypeScript Types
// =============================================================================

export type Tone = z.infer<typeof ToneSchema>;
export type DraftFormValues = z.infer<typeof DraftFormSchema>;
export type EmailDraft = z.infer<typeof EmailDraftSchema>;
export type DraftEmailResponse = z.infer<typeof DraftEmailResponseSchema>;


// =============================================================================
// Helper Validators
// =============================================================================

/**
 * Validates the draft form data.
 * @param data - The data to validate.
 * @returns A discriminated union with either the parsed data or validation errors.
 */
export const validateDraftForm = (data: unknown) => {
  return DraftFormSchema.safeParse(data);
};

/**
 * Validates the API response for a draft email.
 * @param response - The API response to validate.
 * @returns A discriminated union with either the parsed data or validation errors.
 */
export const validateDraftEmailResponse = (response: unknown) => {
  return DraftEmailResponseSchema.safeParse(response);
};
