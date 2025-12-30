/**
 * Utilities to validate/normalize ingestion JSON manifests and payloads.
 */

import { z } from 'zod';

// Define the schema for a single document in the manifest.
export const DocumentSchema = z.object({
  id: z.string().min(1, 'Document ID cannot be empty.'),
  path: z.string().min(1, 'Document path cannot be empty.'),
  data: z.record(z.string(), z.unknown()).optional(),
});

// Define the schema for the ingestion manifest.
export const ManifestSchema = z.object({
  version: z.string().optional(),
  documents: z.array(DocumentSchema),
});

// Infer the TypeScript types from the schemas.
export type Document = z.infer<typeof DocumentSchema>;
export type Manifest = z.infer<typeof ManifestSchema>;

// Legacy aliases
export const IngestionManifestSchema = ManifestSchema;
export type IngestionManifest = Manifest;

/**
 * Validates an ingestion manifest.
 * @param manifest - The manifest to validate.
 * @returns - The validated manifest.
 * @throws - Throws a ZodError if the manifest is invalid.
 */
export const validateManifest = (manifest: unknown): Manifest => {
  return ManifestSchema.parse(manifest);
};

// Legacy alias
export const validateIngestionManifest = validateManifest;

/**
 * Normalizes a document path.
 * @param path - The document path to normalize.
 * @returns - The normalized path.
 */
export const normalizePath = (path: string): string => {
  return path.trim();
};
