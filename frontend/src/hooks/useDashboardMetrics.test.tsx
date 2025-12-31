import { renderHook, waitFor, act } from '@testing-library/react';
import { useDashboardMetrics } from './useDashboardMetrics';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { describe, it, expect, vi, beforeEach, type Mock } from 'vitest';
import { api } from '../lib/api';
import React from 'react';

// Mock the api module
vi.mock('../lib/api', () => ({
  api: {
    fetchStatus: vi.fn(),
    fetchConfig: vi.fn(),
    runDoctor: vi.fn(),
    getIngestionStatus: vi.fn(),
    startIngestion: vi.fn(),
  },
}));

// Create a client for each test
const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false, // Turn off retries for testing
      },
    },
  });

// Wrapper component to provide the QueryClient
const createWrapper = () => {
  const queryClient = createTestQueryClient();
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('useDashboardMetrics', () => {
  beforeEach(() => {
    // Reset mocks before each test
    vi.clearAllMocks();
    // Provide default mocks for queries that run automatically
    (api.fetchStatus as Mock).mockResolvedValue({ status: 'ok', service: 'test', env: 'dev' });
    (api.fetchConfig as Mock).mockResolvedValue({ environment: 'dev', provider: 'test' });
  });

  it('should fetch system status', async () => {
    const mockStatus = { status: 'ok', service: 'test', env: 'dev' };
    (api.fetchStatus as Mock).mockResolvedValue(mockStatus);

    const { result } = renderHook(() => useDashboardMetrics(), { wrapper: createWrapper() });

    await waitFor(() => expect(result.current.systemStatus.isLoading).toBe(false));

    expect(api.fetchStatus).toHaveBeenCalledTimes(1);
    expect(result.current.systemStatus.data).toEqual(mockStatus);
  });

  it('should fetch system config', async () => {
    const mockConfig = { environment: 'dev', provider: 'test' };
    (api.fetchConfig as Mock).mockResolvedValue(mockConfig);

    const { result } = renderHook(() => useDashboardMetrics(), { wrapper: createWrapper() });

    await waitFor(() => expect(result.current.systemConfig.isLoading).toBe(false));

    expect(api.fetchConfig).toHaveBeenCalledTimes(1);
    expect(result.current.systemConfig.data).toEqual(mockConfig);
  });

  it('should not fetch doctor report initially', () => {
    renderHook(() => useDashboardMetrics(), { wrapper: createWrapper() });
    expect(api.runDoctor).not.toHaveBeenCalled();
  });

  it('should fetch doctor report when runCheck is called', async () => {
    const mockReport = { overall_status: 'healthy', checks: [] };
    (api.runDoctor as Mock).mockResolvedValue(mockReport);

    const { result } = renderHook(() => useDashboardMetrics(), { wrapper: createWrapper() });

    // Manually trigger the query
    act(() => {
      result.current.doctorReport.runCheck();
    });

    await waitFor(() => expect(result.current.doctorReport.isLoading).toBe(false));

    expect(api.runDoctor).toHaveBeenCalledTimes(1);
    expect(result.current.doctorReport.data).toEqual(mockReport);
  });

  describe('ingestionStatusQuery', () => {
    it('should not fetch ingestion status if jobId is not provided', () => {
      renderHook(() => useDashboardMetrics(), { wrapper: createWrapper() });
      expect(api.getIngestionStatus).not.toHaveBeenCalled();
    });

    it('should fetch ingestion status if jobId is provided', async () => {
      const jobId = 'test-job-123';
      const mockIngestionStatus = { job_id: jobId, status: 'completed' };
      (api.getIngestionStatus as Mock).mockResolvedValue(mockIngestionStatus);

      const { result } = renderHook(() => useDashboardMetrics(jobId), { wrapper: createWrapper() });

      await waitFor(() => expect(result.current.ingestionStatus.isLoading).toBe(false));

      expect(api.getIngestionStatus).toHaveBeenCalledWith(jobId);
      expect(result.current.ingestionStatus.data).toEqual(mockIngestionStatus);
    });

    it(
      'should poll for ingestion status and stop when completed',
      async () => {
        const jobId = 'test-job-456';
        const runningStatus = { job_id: jobId, status: 'running', folders_processed: 1 };
        const completedStatus = { job_id: jobId, status: 'completed', folders_processed: 2 };

        (api.getIngestionStatus as Mock)
          .mockResolvedValueOnce(runningStatus)
          .mockResolvedValueOnce(completedStatus);

        const { result } = renderHook(() => useDashboardMetrics(jobId), {
          wrapper: createWrapper(),
        });

        await waitFor(() => {
          expect(result.current.ingestionStatus.data).toEqual(runningStatus);
        });

        await waitFor(
          () => {
            expect(result.current.ingestionStatus.data).toEqual(completedStatus);
          },
          { timeout: 6000 },
        ); // Wait for the poll to complete

        // Verify it was called twice and then stopped
        expect(api.getIngestionStatus).toHaveBeenCalledTimes(2);
      },
      { timeout: 10000 },
    );
  });

  describe('startIngestionMutation', () => {
    it('should call startIngestion on mutate', async () => {
      const mockIngestionResponse = { job_id: 'new-job-123', status: 'started' };
      (api.startIngestion as Mock).mockResolvedValue(mockIngestionResponse);
      const { result } = renderHook(() => useDashboardMetrics(), { wrapper: createWrapper() });

      await act(async () => {
        await result.current.startIngestion.mutateAsync({ prefix: 'Outlook/' });
      });

      expect(api.startIngestion).toHaveBeenCalledWith('Outlook/', undefined, undefined);
      await waitFor(() => {
        expect(result.current.startIngestion.data).toEqual(mockIngestionResponse);
      });
    });
  });
});
