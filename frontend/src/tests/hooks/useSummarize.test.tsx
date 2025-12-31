import { renderHook, waitFor } from '@testing-library/react';
import { useSummarize } from '../../hooks/useSummarize';
import { api } from '../../lib/api';
import { describe, it, expect, vi, beforeEach, type Mock } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';

// Mock the api module
vi.mock('../../lib/api');

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      mutations: {
        retry: false,
      },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('useSummarize', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('should call summarizeThread and return data on success', async () => {
    const mockData = {
      correlation_id: '123',
      summary: {
        summary: 'This is a summary.',
        key_points: ['Point 1', 'Point 2'],
        action_items: ['Item 1'],
      },
    };
    (api.summarizeThread as Mock).mockResolvedValue(mockData);

    const { result } = renderHook(() => useSummarize(), {
      wrapper: createWrapper(),
    });

    result.current.summarize({ threadId: 'thread-123' });

    await waitFor(() => {
      expect(result.current.status).toBe('success');
    });

    expect(api.summarizeThread).toHaveBeenCalledWith('thread-123', undefined);
    expect(result.current.data).toEqual(mockData);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBe(null);
  });

  it('should handle errors from the api', async () => {
    const mockError = new Error('Summarization failed');
    (api.summarizeThread as Mock).mockRejectedValue(mockError);

    const { result } = renderHook(() => useSummarize(), {
      wrapper: createWrapper(),
    });

    result.current.summarize({ threadId: 'thread-456' });

    await waitFor(() => {
      expect(result.current.status).toBe('error');
    });

    expect(api.summarizeThread).toHaveBeenCalledWith('thread-456', undefined);
    expect(result.current.data).toBe(undefined);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toEqual(mockError);
  });
});
