/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';

import { api } from '@/lib/api';
import { useSearch } from './useSearch';

vi.mock('@/lib/api');

const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
    },
  },
});

describe('useSearch', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = createTestQueryClient();
  });

  afterEach(() => {
    vi.clearAllMocks();
    queryClient.clear();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  it('should return loading state initially', () => {
    // To test loading state, we need a query that never resolves.
    vi.mocked(api.search).mockReturnValue(new Promise(() => {}));
    const { result } = renderHook(() => useSearch({ query: 'test' }), { wrapper });
    expect(result.current.isLoading).toBe(true);
  });

  it('should return data on successful fetch', async () => {
    const mockData = {
      results: [{ chunk_id: '1', conversation_id: '1', content: 'Test content', score: 1 }],
      total_count: 1,
      query_time_ms: 100,
    };
    vi.mocked(api.search).mockResolvedValue(mockData);

    const { result } = renderHook(() => useSearch({ query: 'test' }), { wrapper });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.data).toEqual(mockData);
    expect(result.current.error).toBe(null);
  });

  it('should return an error when fetch fails', async () => {
    const mockError = new Error('Failed to fetch');
    vi.mocked(api.search).mockRejectedValue(mockError);

    const { result } = renderHook(() => useSearch({ query: 'test' }), { wrapper });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.error).toEqual(mockError);
    expect(result.current.data).toBeUndefined();
  });

  it('should not fetch when query is empty', () => {
    const { result } = renderHook(() => useSearch({ query: '' }), { wrapper });
    expect(result.current.isLoading).toBe(false);
    expect(api.search).not.toHaveBeenCalled();
  });

  it('should not fetch when disabled', () => {
    const { result } = renderHook(() => useSearch({ query: 'test', enabled: false }), { wrapper });
    expect(result.current.isLoading).toBe(false);
    expect(api.search).not.toHaveBeenCalled();
  });
});
