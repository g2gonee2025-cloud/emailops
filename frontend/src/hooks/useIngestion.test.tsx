/**
 * @vitest-environment jsdom
 */
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { useIngestion } from './useIngestion';
import { api } from '@/lib/api';
import React from 'react';

// Mock the api module
vi.mock('@/lib/api', () => ({
  api: {
    listS3Folders: vi.fn(),
    startIngestion: vi.fn(),
    getIngestionStatus: vi.fn(),
  },
}));

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('useIngestion', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should fetch S3 folders', async () => {
    const mockFolders = { prefix: 'Outlook/', folders: [{ folder: 'folder1' }], count: 1 };
    (api.listS3Folders as vi.Mock).mockResolvedValue(mockFolders);

    const { result } = renderHook(() => useIngestion(), { wrapper: createWrapper() });

    await waitFor(() => expect(result.current.isLoadingFolders).toBe(false));

    expect(result.current.folders).toEqual(mockFolders);
    expect(api.listS3Folders).toHaveBeenCalledTimes(1);
  });

  it('should handle errors when fetching S3 folders', async () => {
    const mockError = new Error('Failed to fetch folders');
    (api.listS3Folders as vi.Mock).mockRejectedValue(mockError);

    const { result } = renderHook(() => useIngestion(), { wrapper: createWrapper() });

    await waitFor(() => expect(result.current.isLoadingFolders).toBe(false));

    expect(result.current.foldersError).toEqual(mockError);
  });

  it('should start ingestion successfully', async () => {
    const mockJob = { job_id: '123', status: 'started', message: 'Ingestion started' };
    (api.startIngestion as vi.Mock).mockResolvedValue(mockJob);

    const { result } = renderHook(() => useIngestion(), { wrapper: createWrapper() });

    result.current.startIngestion({ prefix: 'test/', dryRun: true });

    await waitFor(() => expect(result.current.isStartingIngestion).toBe(false));

    expect(api.startIngestion).toHaveBeenCalledWith('test/', undefined, true);
    expect(result.current.startIngestionError).toBe(null);
  });

  it('should handle errors when starting ingestion', async () => {
    const mockError = new Error('Failed to start ingestion');
    (api.startIngestion as vi.Mock).mockRejectedValue(mockError);

    const { result } = renderHook(() => useIngestion(), { wrapper: createWrapper() });

    result.current.startIngestion({});

    await waitFor(() => expect(result.current.isStartingIngestion).toBe(false));

    expect(result.current.startIngestionError).toEqual(mockError);
  });

  it('should poll for ingestion status', async () => {
    const jobId = 'job-123';
    const mockStatus1 = { job_id: jobId, status: 'running', message: 'Processing...' };
    const mockStatus2 = { job_id: jobId, status: 'completed', message: 'Done' };

    const getIngestionStatusMock = api.getIngestionStatus as vi.Mock;
    getIngestionStatusMock
      .mockResolvedValueOnce(mockStatus1)
      .mockResolvedValue(mockStatus2);

    const { result } = renderHook(() => {
      const { useIngestionStatus } = useIngestion();
      return useIngestionStatus(jobId, true, 100);
    }, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(getIngestionStatusMock).toHaveBeenCalledTimes(1);
      expect(result.current.data).toEqual(mockStatus1);
    });

    await waitFor(() => {
      expect(getIngestionStatusMock).toHaveBeenCalledTimes(2);
      expect(result.current.data).toEqual(mockStatus2);
    });
  });
});
