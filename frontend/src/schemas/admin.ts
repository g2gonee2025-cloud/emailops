import { z } from 'zod';

// Schema for an individual doctor check
export const DoctorCheckSchema = z.object({
  name: z.string(),
  status: z.enum(['pass', 'fail', 'warn']),
  message: z.string().optional(),
  details: z.record(z.unknown()).optional(),
});

// Schema for the overall doctor report
export const DoctorReportSchema = z.object({
  overall_status: z.enum(['healthy', 'degraded', 'unhealthy']),
  checks: z.array(DoctorCheckSchema),
});

// Schema for the environment status data
export const StatusDataSchema = z.object({
    env: z.string(),
    service: z.string(),
    status: z.string(),
});

// Inferred TypeScript types
export type DoctorCheck = z.infer<typeof DoctorCheckSchema>;
export type DoctorReport = z.infer<typeof DoctorReportSchema>;
export type StatusData = z.infer<typeof StatusDataSchema>;

const SENSITIVE_KEY_SUBSTRINGS = ['KEY', 'SECRET', 'TOKEN', 'URL', 'PASS', 'CONNECTION_STRING', 'PW'];

/**
 * Redacts sensitive values in an object based on key substrings.
 * @param obj The object to redact.
 * @returns A new object with sensitive values replaced, or null if the input was null/undefined.
 */
export const redactObject = (obj: Record<string, unknown> | null | undefined): Record<string, unknown> | null => {
    if (!obj) return null;

    return Object.fromEntries(
        Object.entries(obj).map(([key, value]) => {
            const upperKey = key.toUpperCase();
            if (SENSITIVE_KEY_SUBSTRINGS.some(substring => upperKey.includes(substring))) {
                return [key, '******'];
            }
            return [key, value];
        })
    );
};
