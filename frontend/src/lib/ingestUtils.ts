/**
 * Ingestion manifest and payload utilities.
 */

import { z } from 'zod';

// Define the schema for a single document in the manifest.
export const DocumentSchema = z.object({
  id: z.string().min(1, 'Document ID cannot be empty.'),
  path: z.string().min(1, 'Document path cannot be empty.'),
});

// Define the schema for the ingestion manifest.
export const ManifestSchema = z.object({
  documents: z.array(DocumentSchema),
});

// Infer the TypeScript types from the schemas.
export type Document = z.infer<typeof DocumentSchema>;
export type Manifest = z.infer<typeof ManifestSchema>;

/**
 * Validates an ingestion manifest.
 *
 * @param manifest - The manifest to validate.
 * @returns - The validated manifest.
 * @throws - Throws a ZodError if the manifest is invalid.
 */
export const validateManifest = (manifest: unknown): Manifest => {
  return ManifestSchema.parse(manifest);
};

/**
 * Normalizes a document path.
 * For now, this is a placeholder.
 *
 * @param path - The document path to normalize.
 * @returns - The normalized path.
 */
export const normalizePath = (path: string): string => {
  return path.trim();
};
