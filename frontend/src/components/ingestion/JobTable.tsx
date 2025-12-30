import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/Table";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Job } from "@/schemas/job";

type JobTableProps = {
  jobs: Job[];
  onRetry: (jobId: string) => void;
};

export function JobTable({ jobs, onRetry }: JobTableProps) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Job ID</TableHead>
          <TableHead>Status</TableHead>
          <TableHead>Created At</TableHead>
          <TableHead>Actions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {jobs.map((job) => (
          <TableRow key={job.id}>
            <TableCell>{job.id}</TableCell>
            <TableCell>
              <Badge
                variant={
                  job.status === "succeeded"
                    ? "default"
                    : job.status === "failed"
                    ? "destructive"
                    : "outline"
                }
              >
                {job.status}
              </Badge>
            </TableCell>
            <TableCell>{new Date(job.createdAt).toLocaleString()}</TableCell>
            <TableCell>
              {job.status === "failed" && (
                <Button onClick={() => onRetry(job.id)}>Retry</Button>
              )}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
