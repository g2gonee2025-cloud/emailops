import { z } from 'zod';

/**
 * Schema for the health response from the API.
 * This validates the structure of the data received from the /health endpoint.
 */
export const HealthResponseSchema = z.object({
  status: z.string().min(1, 'Status cannot be empty'),
  version: z.string().min(1, 'Version cannot be empty'),
  environment: z.string().min(1, 'Environment cannot be empty'),
});

// Infer the TypeScript type from the HealthResponseSchema
export type HealthResponse = z.infer<typeof HealthResponseSchema>;

/**
 * Schema for a single log entry in the dashboard's live process stream.
 * This ensures log objects have a consistent structure.
 */
export const LogEntrySchema = z.object({
  logId: z.number(),
  timestamp: z.string(),
  message: z.string(),
  level: z.enum(['INFO', 'WARN', 'ERROR', 'DEBUG']),
});

// Infer the TypeScript type from the LogEntrySchema
export type LogEntry = z.infer<typeof LogEntrySchema>;

/**
 * Helper schema for validating an array of log entries.
 */
export const LogEntriesSchema = z.array(LogEntrySchema);

// Infer the TypeScript type for an array of log entries
export type LogEntries = z.infer<typeof LogEntriesSchema>;
