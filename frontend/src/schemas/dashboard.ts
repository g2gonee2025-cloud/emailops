import { z } from 'zod';

/**
 * @description Schema for individual log entries streamed to the dashboard.
 * Ensures that each log has a numeric ID, a timestamp, a message, and a valid log level.
 */
export const LogEntrySchema = z.object({
  logId: z.number(),
  timestamp: z.string().min(1, 'Timestamp is required.'),
  message: z.string().min(1, 'Log message is required.'),
  level: z.enum(['INFO', 'WARN', 'ERROR', 'DEBUG']),
});

/**
 * @description Schema for an array of log entries.
 */
export const LogEntriesSchema = z.array(LogEntrySchema);

/**
 * @description Schema for the overall health response from the API.
 * Validates the service's status, version, and environment.
 */
export const HealthResponseSchema = z.object({
  status: z.string().min(1, 'Status is required.'),
  version: z.string().min(1, 'Version is required.'),
  environment: z.string().min(1, 'Environment is required.'),
});

// Inferred TypeScript types
export type LogEntry = z.infer<typeof LogEntrySchema>;
export type LogEntries = z.infer<typeof LogEntriesSchema>;
export type HealthResponse = z.infer<typeof HealthResponseSchema>;
