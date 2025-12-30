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
 * Redacts sensitive keys in a nested object.
 * @param obj The object to redact.
 * @returns A new object with sensitive keys redacted.
 */
export const redactObject = (obj: unknown): unknown => {
    if (typeof obj !== 'object' || obj === null) {
      return obj;
    }

    if (Array.isArray(obj)) {
      return obj.map(redactObject);
    }

    const newObj: Record<string, unknown> = {};
    for (const key in obj) {
      if (Object.prototype.hasOwnProperty.call(obj, key)) {
        const lowerKey = key.toLowerCase();
        if (lowerKey.includes('key') || lowerKey.includes('secret') || lowerKey.includes('token') || lowerKey.includes('password')) {
          newObj[key] = '********';
        } else {
          newObj[key] = redactObject((obj as Record<string, unknown>)[key]);
        }
      }
    }
    return newObj;
  };


/**
 * Schema for the application configuration.
 * This is a placeholder and should be updated with the actual config structure.
 */
export const ConfigSchema = z.object({
    // Example fields - replace with actual config fields
    api_url: z.string().url().optional(),
    log_level: z.enum(['DEBUG', 'INFO', 'WARNING', 'ERROR']).optional(),
    max_pool_size: z.coerce.number().int().positive().optional(),
  });

// Inferred TypeScript type for the configuration
export type AppConfig = z.infer<typeof ConfigSchema>;
