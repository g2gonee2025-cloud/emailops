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
