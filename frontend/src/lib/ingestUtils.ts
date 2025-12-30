/**
 * Utilities to validate/normalize ingestion JSON manifests and payloads.
 */

import { z } from 'zod';

// Define a basic schema for an ingestion manifest.
export const IngestionManifestSchema = z.object({
  version: z.string(),
  items: z.array(z.object({
    id: z.string(),
    data: z.record(z.string(), z.unknown()),
  })),
});

export type IngestionManifest = z.infer<typeof IngestionManifestSchema>;

/**
 * Validates an ingestion manifest against the schema.
 * @param manifest The manifest to validate.
 * @returns The validated manifest.
 * @throws ZodError if the manifest is invalid.
 */
export function validateIngestionManifest(manifest: unknown): IngestionManifest {
  return IngestionManifestSchema.parse(manifest);
}
