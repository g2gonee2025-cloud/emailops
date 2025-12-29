/**
 * @vitest-environment jsdom
 */
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { describe, expect, it, vi } from 'vitest';
import { api } from '../lib/api';
import { useDashboardMetrics } from './useDashboardMetrics';

// Mock the api module
vi.mock('../lib/api');

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
  // eslint-disable-next-line react/display-name
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('useDashboardMetrics', () => {
  it('should return dashboard metrics data', async () => {
    const mockMetrics = {
      total_conversations: 100,
      total_threads: 200,
      total_messages: 500,
      total_documents: 50,
      total_chunks: 1000,
      total_embeddings: 1000,
      storage_used_gb: 10,
      avg_response_time_ms: 150,
    };

    // Use vi.mocked to properly type the mocked function
    vi.mocked(api.getDashboardMetrics).mockResolvedValue(mockMetrics);

    const { result } = renderHook(() => useDashboardMetrics(), {
      wrapper: createWrapper(),
    });

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.metrics).toEqual(mockMetrics);
    expect(result.current.error).toBe(null);
  });

  it('should handle errors', async () => {
    const mockError = new Error('Failed to fetch metrics');
    vi.mocked(api.getDashboardMetrics).mockRejectedValue(mockError);

    const { result } = renderHook(() => useDashboardMetrics(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.metrics).toBeUndefined();
    expect(result.current.error).toEqual(mockError);
  });
});
