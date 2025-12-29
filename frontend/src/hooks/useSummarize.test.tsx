/** @vitest-environment jsdom */
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { describe, it, expect, vi } from 'vitest';
import { useSummarize } from './useSummarize';
import { api } from '../lib/api';
import React from 'react';

// Mock the api module
vi.mock('../lib/api');

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
    },
  },
});

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
);

describe('useSummarize', () => {
  it('should call summarizeThread and return data on success', async () => {
    const mockData = {
      correlation_id: '123',
      summary: {
        summary: 'This is a summary.',
        key_points: ['Point 1', 'Point 2'],
        action_items: ['Item 1'],
      },
    };
    vi.mocked(api.summarizeThread).mockResolvedValue(mockData);

    const { result } = renderHook(() => useSummarize(), { wrapper });

    result.current.summarize({ threadId: 'thread-123' });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockData);
    expect(api.summarizeThread).toHaveBeenCalledWith('thread-123', undefined);
  });

  it('should handle errors', async () => {
    const mockError = new Error('Failed to summarize');
    vi.mocked(api.summarizeThread).mockRejectedValue(mockError);

    const { result } = renderHook(() => useSummarize(), { wrapper });

    result.current.summarize({ threadId: 'thread-456' });

    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(result.current.error).toBe(mockError);
  });
});
