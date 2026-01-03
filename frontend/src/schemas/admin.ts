import { z } from 'zod';

// Schema for an individual health check
export const doctorCheckSchema = z.object({
  name: z.string(),
  status: z.enum(['pass', 'fail', 'warn']),
  message: z.string().optional(),
  details: z.record(z.unknown()).optional(),
});

// Schema for the overall doctor report
export const doctorReportSchema = z.object({
  overall_status: z.enum(['healthy', 'degraded', 'unhealthy']),
  checks: z.array(doctorCheckSchema),
});

// Schema for the basic server status
export const statusDataSchema = z.object({
    env: z.string(),
    service: z.string(),
    status: z.string(),
});


// Inferred TypeScript types
export type DoctorCheck = z.infer<typeof doctorCheckSchema>;
export type DoctorReport = z.infer<typeof doctorReportSchema>;
export type StatusData = z.infer<typeof statusDataSchema>;

/**
 * Redacts sensitive information from an object recursively.
 * @param obj The object or array to redact.
 * @returns A new object or array with sensitive values replaced.
 */
export function redactObject<T>(obj: T): T {
  if (typeof obj !== 'object' || obj === null) {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map(redactObject) as unknown as T;
  }

  const SENSITIVE_KEYWORDS = ['key', 'secret', 'token', 'password', 'url', 'uri', 'host'];
  const redacted = { ...obj } as Record<string, unknown>;

  for (const key in redacted) {
    if (Object.prototype.hasOwnProperty.call(redacted, key)) {
      if (SENSITIVE_KEYWORDS.some(k => key.toLowerCase().includes(k))) {
        redacted[key] = '••••••••';
      } else {
        redacted[key] = redactObject(redacted[key]);
      }
    }
  }

  return redacted as unknown as T;
}

/**
 * Schema for the application configuration.
 * Validates configuration fields returned from the backend API.
 */
export const ConfigSchema = z.object({
    api_url: z.string().url().optional(),
    log_level: z.enum(['DEBUG', 'INFO', 'WARNING', 'ERROR']).optional(),
    max_pool_size: z.coerce.number().int().positive().optional(),
  });

// Inferred TypeScript type for the configuration
export type AppConfig = z.infer<typeof ConfigSchema>;
