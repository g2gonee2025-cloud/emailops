import { describe, it, expect } from 'vitest';
import { sortByTimestamp, flattenThreads } from './threadUtils';
import type { SearchResult } from './api';

describe('threadUtils', () => {
  describe('sortByTimestamp', () => {
    it('should sort search results by timestamp in ascending order', () => {
      const items: SearchResult[] = [
        {
          chunk_id: '1',
          conversation_id: 'a',
          content: '',
          score: 1,
          metadata: { timestamp: '2023-01-02T00:00:00Z' },
        },
        {
          chunk_id: '2',
          conversation_id: 'a',
          content: '',
          score: 1,
          metadata: { timestamp: '2023-01-01T00:00:00Z' },
        },
        {
          chunk_id: '3',
          conversation_id: 'a',
          content: '',
          score: 1,
          metadata: { timestamp: '2023-01-03T00:00:00Z' },
        },
      ];

      const sorted = sortByTimestamp(items, 'asc');
      expect(sorted[0].metadata?.timestamp).toBe('2023-01-01T00:00:00Z');
      expect(sorted[1].metadata?.timestamp).toBe('2023-01-02T00:00:00Z');
      expect(sorted[2].metadata?.timestamp).toBe('2023-01-03T00:00:00Z');
    });

    it('should sort search results by timestamp in descending order', () => {
      const items: SearchResult[] = [
        {
          chunk_id: '1',
          conversation_id: 'a',
          content: '',
          score: 1,
          metadata: { timestamp: '2023-01-02T00:00:00Z' },
        },
        {
          chunk_id: '2',
          conversation_id: 'a',
          content: '',
          score: 1,
          metadata: { timestamp: '2023-01-01T00:00:00Z' },
        },
        {
          chunk_id: '3',
          conversation_id: 'a',
          content: '',
          score: 1,
          metadata: { timestamp: '2023-01-03T00:00:00Z' },
        },
      ];

      const sorted = sortByTimestamp(items, 'desc');
      expect(sorted[0].metadata?.timestamp).toBe('2023-01-03T00:00:00Z');
      expect(sorted[1].metadata?.timestamp).toBe('2023-01-02T00:00:00Z');
      expect(sorted[2].metadata?.timestamp).toBe('2023-01-01T00:00:00Z');
    });

    it('should handle items with missing timestamps gracefully', () => {
      const items: SearchResult[] = [
        {
          chunk_id: '1',
          conversation_id: 'a',
          content: '',
          score: 1,
          metadata: { timestamp: '2023-01-02T00:00:00Z' },
        },
        { chunk_id: '2', conversation_id: 'a', content: '', score: 1 },
        {
          chunk_id: '3',
          conversation_id: 'a',
          content: '',
          score: 1,
          metadata: { timestamp: '2023-01-01T00:00:00Z' },
        },
      ];

      const sorted = sortByTimestamp(items);
      // The order of items with missing timestamps is not guaranteed,
      // but the ones with timestamps should be sorted correctly.
      const timestamps = sorted
        .map((item) => item.metadata?.timestamp)
        .filter(Boolean);
      expect(timestamps).toEqual([
        '2023-01-01T00:00:00Z',
        '2023-01-02T00:00:00Z',
      ]);
    });
  });

  describe('flattenThreads', () => {
    it('should flatten an array of threads into a single array', () => {
      const threads = [
        [1, 2, 3],
        [4, 5],
        [6, 7, 8],
      ];
      const flattened = flattenThreads(threads);
      expect(flattened).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    });

    it('should handle an empty array of threads', () => {
      const threads: number[][] = [];
      const flattened = flattenThreads(threads);
      expect(flattened).toEqual([]);
    });

    it('should handle threads with a single item', () => {
      const threads = [[1], [2], [3]];
      const flattened = flattenThreads(threads);
      expect(flattened).toEqual([1, 2, 3]);
    });
  });
});
