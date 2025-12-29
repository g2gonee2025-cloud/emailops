import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { useThread } from './useThread';
import { api, SummarizeResponse, Thread } from '@/lib/api';
import React from 'react';

// Mock the api module
vi.mock('@/lib/api', async (importOriginal) => {
  const mod = await importOriginal<typeof import('@/lib/api')>();
  return {
    ...mod,
    api: {
      ...mod.api,
      fetchThread: vi.fn(),
      summarizeThread: vi.fn(),
    },
  };
});

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

describe('useThread', () => {
  const threadId = 'test-thread-123';
  const mockThread: Thread = {
    thread_id: threadId,
    subject: 'Test Subject',
    participants: ['a@test.com', 'b@test.com'],
    messages: [],
  };

  beforeEach(() => {
    vi.resetAllMocks();
  });

  describe('Querying', () => {
    it('should be in a loading state initially and then return data on success', async () => {
      (api.fetchThread as vi.Mock).mockResolvedValue(mockThread);

      const { result } = renderHook(() => useThread(threadId), {
        wrapper: createWrapper(),
      });

      expect(result.current.isLoading).toBe(true);
      expect(result.current.data).toBeUndefined();

      await waitFor(() => expect(result.current.isLoading).toBe(false));

      expect(result.current.data).toEqual(mockThread);
      expect(api.fetchThread).toHaveBeenCalledWith(threadId);
    });

    it('should not fetch data if threadId is null', () => {
      const { result } = renderHook(() => useThread(null), {
        wrapper: createWrapper(),
      });

      expect(result.current.isLoading).toBe(false);
      expect(result.current.data).toBeUndefined();
      expect(api.fetchThread).not.toHaveBeenCalled();
    });

    it('should handle query errors', async () => {
      const testError = new Error('Failed to fetch');
      (api.fetchThread as vi.Mock).mockRejectedValue(testError);

      const { result } = renderHook(() => useThread(threadId), {
        wrapper: createWrapper(),
      });

      await waitFor(() => expect(result.current.isLoading).toBe(false));

      expect(result.current.error).toBe(testError);
      expect(result.current.data).toBeUndefined();
    });
  });

  describe('Summarize Mutation', () => {
    const mockSummary: SummarizeResponse = {
      summary: { summary: 'This is a test summary.' },
    };

    it('should call api.summarizeThread and update state on success', async () => {
      (api.summarizeThread as vi.Mock).mockResolvedValue(mockSummary);
      (api.fetchThread as vi.Mock).mockResolvedValue(mockThread); // Ensure the query part is settled

      const { result } = renderHook(() => useThread(threadId), {
        wrapper: createWrapper(),
      });

      // Wait for the initial query to finish before mutating
      await waitFor(() => expect(result.current.isLoading).toBe(false));

      result.current.summarize();

      await waitFor(() => expect(result.current.isSummarizing).toBe(false));

      expect(result.current.summaryData).toEqual(mockSummary);
      expect(api.summarizeThread).toHaveBeenCalledWith(threadId, undefined);
    });

    it('should handle summarize errors', async () => {
      const testError = new Error('Summarization failed');
      (api.summarizeThread as vi.Mock).mockRejectedValue(testError);
      (api.fetchThread as vi.Mock).mockResolvedValue(mockThread); // Ensure the query part is settled

      const { result } = renderHook(() => useThread(threadId), {
        wrapper: createWrapper(),
      });

      await waitFor(() => expect(result.current.isLoading).toBe(false));

      result.current.summarize();

      await waitFor(() => expect(result.current.isSummarizing).toBe(false));

      expect(result.current.summarizeError).toBe(testError);
    });
  });
});