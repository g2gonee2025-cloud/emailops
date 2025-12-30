import { z } from "zod";

export const JobSchema = z.object({
  id: z.string(),
  status: z.enum(["pending", "running", "succeeded", "failed"]),
  createdAt: z.string(),
});

export type Job = z.infer<typeof JobSchema>;
