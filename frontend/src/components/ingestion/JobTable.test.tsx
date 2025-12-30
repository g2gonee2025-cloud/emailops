import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { JobTable } from "./JobTable";

const mockJobs = [
  {
    id: "job-1",
    status: "succeeded" as const,
    createdAt: "2024-01-01T12:00:00Z",
  },
  {
    id: "job-2",
    status: "failed" as const,
    createdAt: "2024-01-01T13:00:00Z",
  },
  {
    id: "job-3",
    status: "pending" as const,
    createdAt: "2024-01-01T14:00:00Z",
  },
  {
    id: "job-4",
    status: "running" as const,
    createdAt: "2024-01-01T15:00:00Z",
  },
];

describe("JobTable", () => {
  it("renders the table with job data", () => {
    const onRetry = vi.fn();
    render(<JobTable jobs={mockJobs} onRetry={onRetry} />);

    // Check for headers
    expect(screen.getByText("Job ID")).toBeInTheDocument();
    expect(screen.getByText("Status")).toBeInTheDocument();
    expect(screen.getByText("Created At")).toBeInTheDocument();
    expect(screen.getByText("Actions")).toBeInTheDocument();

    // Check for job data
    expect(screen.getByText("job-1")).toBeInTheDocument();
    expect(screen.getByText("succeeded")).toBeInTheDocument();
    expect(screen.getByText("job-2")).toBeInTheDocument();
    expect(screen.getByText("failed")).toBeInTheDocument();
  });

  it("shows a retry button for failed jobs and calls onRetry when clicked", () => {
    const onRetry = vi.fn();
    render(<JobTable jobs={mockJobs} onRetry={onRetry} />);

    const retryButton = screen.getByRole("button", { name: "Retry" });
    expect(retryButton).toBeInTheDocument();

    fireEvent.click(retryButton);
    expect(onRetry).toHaveBeenCalledWith("job-2");
  });

  it("does not show a retry button for non-failed jobs", () => {
    const onRetry = vi.fn();
    const nonFailedJobs = mockJobs.filter((job) => job.status !== "failed");
    render(<JobTable jobs={nonFailedJobs} onRetry={onRetry} />);

    const retryButton = screen.queryByRole("button", { name: "Retry" });
    expect(retryButton).not.toBeInTheDocument();
  });
});
