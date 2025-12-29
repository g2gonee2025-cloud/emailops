import { z } from 'zod';
import type { PushDocument } from '../lib/api';

/**
 * Schema for a single document in the manual upload form.
 * Text content is required.
 * Metadata is an optional string that must be valid JSON if provided.
 */
export const ManualDocumentSchema = z.object({
  text: z.string().min(1, 'Document text cannot be empty.'),
  metadata: z.string().refine((val) => {
    if (val === '') return true; // Allow empty string
    try {
      JSON.parse(val);
      return true;
    } catch {
      return false;
    }
  }, { message: 'Metadata must be a valid JSON string.' }),
});

/**
 * Schema for the complete manual upload form, containing an array of documents.
 */
export const ManualUploadFormSchema = z.object({
  documents: z.array(ManualDocumentSchema),
});

// Inferred TypeScript types
export type ManualDocument = z.infer<typeof ManualDocumentSchema>;
export type ManualUploadForm = z.infer<typeof ManualUploadFormSchema>;

/**
 * Transforms raw form data into the PushDocument array for the API.
 * This function assumes the data has already been validated.
 * @param formData The validated form data.
 * @returns An array of PushDocument objects ready for the API.
 */
export const transformToPushDocuments = (formData: ManualUploadForm): PushDocument[] => {
  return formData.documents
    .filter(doc => doc.text.trim().length > 0) // Filter out empty text docs
    .map(doc => ({
      text: doc.text.trim(),
      metadata: doc.metadata ? JSON.parse(doc.metadata) : {},
    }));
};
