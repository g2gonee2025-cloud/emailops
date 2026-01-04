import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { api, type IngestStatusResponse } from '../lib/api';
import { useIngestionStatus } from './useIngestion';
import React from 'react';
import { describe, it, expect, vi, afterEach, type Mock } from 'vitest';

vi.mock('../lib/api', () => ({
  api: {
    getIngestionStatus: vi.fn(),
    listS3Folders: vi.fn(),
    startIngestion: vi.fn(),
  },
}));

const mockedApi = api as { getIngestionStatus: Mock };

const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={createTestQueryClient()}>{children}</QueryClientProvider>
);

describe('useIngestionStatus', () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it('should not fetch when jobId is null', () => {
    renderHook(() => useIngestionStatus(null), { wrapper });
    expect(mockedApi.getIngestionStatus).not.toHaveBeenCalled();
  });

  it('should fetch status when jobId is provided', async () => {
    const mockResponse: IngestStatusResponse = {
      job_id: 'test-job-123',
      status: 'processing',
      folders_processed: 5,
      threads_created: 10,
      chunks_created: 50,
      embeddings_generated: 50,
      errors: 0,
      skipped: 0,
      message: 'Processing...',
    };
    mockedApi.getIngestionStatus.mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useIngestionStatus('test-job-123'), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(mockedApi.getIngestionStatus).toHaveBeenCalledWith(
      'test-job-123',
      expect.objectContaining({ signal: expect.any(AbortSignal) })
    );
    expect(result.current.data).toEqual(mockResponse);
    expect(result.current.error).toBe(null);
  });

  it('should return error when API call fails', async () => {
    const mockError = new Error('API Error');
    mockedApi.getIngestionStatus.mockRejectedValue(mockError);

    const { result } = renderHook(() => useIngestionStatus('test-job-123'), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBe(mockError);
    expect(result.current.data).toBe(undefined);
  });

  it('should return data with terminal status "completed"', async () => {
    const mockResponse: IngestStatusResponse = {
      job_id: 'test-job-123',
      status: 'completed',
      folders_processed: 10,
      threads_created: 20,
      chunks_created: 100,
      embeddings_generated: 100,
      errors: 0,
      skipped: 0,
      message: 'Completed successfully',
    };
    mockedApi.getIngestionStatus.mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useIngestionStatus('test-job-123'), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data?.status).toBe('completed');
  });

  it('should return data with terminal status "failed"', async () => {
    const mockResponse: IngestStatusResponse = {
      job_id: 'test-job-123',
      status: 'failed',
      folders_processed: 5,
      threads_created: 10,
      chunks_created: 50,
      embeddings_generated: 50,
      errors: 3,
      skipped: 0,
      message: 'Job failed',
    };
    mockedApi.getIngestionStatus.mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useIngestionStatus('test-job-123'), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data?.status).toBe('failed');
  });

  it('should return data with terminal status "completed_with_errors"', async () => {
    const mockResponse: IngestStatusResponse = {
      job_id: 'test-job-123',
      status: 'completed_with_errors',
      folders_processed: 10,
      threads_created: 18,
      chunks_created: 90,
      embeddings_generated: 90,
      errors: 2,
      skipped: 1,
      message: 'Completed with some errors',
    };
    mockedApi.getIngestionStatus.mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useIngestionStatus('test-job-123'), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data?.status).toBe('completed_with_errors');
  });

  it('should expose refetch function', async () => {
    const mockResponse: IngestStatusResponse = {
      job_id: 'test-job-123',
      status: 'completed',
      folders_processed: 10,
      threads_created: 20,
      chunks_created: 100,
      embeddings_generated: 100,
      errors: 0,
      skipped: 0,
      message: 'Completed',
    };
    mockedApi.getIngestionStatus.mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useIngestionStatus('test-job-123'), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(typeof result.current.refetch).toBe('function');
  });

  it('should return all expected properties', async () => {
    const mockResponse: IngestStatusResponse = {
      job_id: 'test-job-123',
      status: 'processing',
      folders_processed: 5,
      threads_created: 10,
      chunks_created: 50,
      embeddings_generated: 50,
      errors: 0,
      skipped: 0,
      message: 'Processing...',
    };
    mockedApi.getIngestionStatus.mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useIngestionStatus('test-job-123'), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current).toHaveProperty('data');
    expect(result.current).toHaveProperty('isLoading');
    expect(result.current).toHaveProperty('error');
    expect(result.current).toHaveProperty('refetch');
  });
});
