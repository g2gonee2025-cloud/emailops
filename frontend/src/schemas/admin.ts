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
 * @param data The object to redact.
 * @returns A new object with sensitive values replaced.
 */
export const redactObject = (data: unknown): unknown => {
    if (typeof data !== 'object' || data === null) {
      return data;
    }

    if (Array.isArray(data)) {
      return data.map(redactObject);
    }

    const SENSITIVE_KEYS = ['key', 'secret', 'token', 'password', 'url', 'uri', 'host'];

    return Object.fromEntries(
      Object.entries(data).map(([key, value]) => {
        if (SENSITIVE_KEYS.some(k => key.toLowerCase().includes(k))) {
          return [key, '********'];
        }
        return [key, redactObject(value)];
      })
    );
  };
