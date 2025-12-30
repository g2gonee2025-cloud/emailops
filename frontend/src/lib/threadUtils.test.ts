import { describe, it, expect } from 'vitest';
import { sortThreadChunks, flattenThreadContent } from './threadUtils';
import type { Thread } from './threadUtils';

const createMockChunk = (chunk_id: string, score: number, content: string): Thread[number] => ({
  chunk_id,
  score,
  content,
  conversation_id: `conv-${chunk_id}`,
});

describe('threadUtils', () => {
  describe('sortThreadChunks', () => {
    it('should sort chunks by score in descending order', () => {
      const thread: Thread = [
        createMockChunk('c1', 0.8, 'Content 1'),
        createMockChunk('c2', 0.9, 'Content 2'),
        createMockChunk('c3', 0.7, 'Content 3'),
      ];
      const sorted = sortThreadChunks(thread);
      expect(sorted.map(c => c.chunk_id)).toEqual(['c2', 'c1', 'c3']);
    });

    it('should maintain stable sort using chunk_id when scores are equal', () => {
      const thread: Thread = [
        createMockChunk('c2', 0.9, 'Content 2'),
        createMockChunk('c1', 0.9, 'Content 1'),
        createMockChunk('c3', 0.7, 'Content 3'),
      ];
      const sorted = sortThreadChunks(thread);
      expect(sorted.map(c => c.chunk_id)).toEqual(['c1', 'c2', 'c3']);
    });

    it('should handle an empty thread', () => {
      const thread: Thread = [];
      const sorted = sortThreadChunks(thread);
      expect(sorted).toEqual([]);
    });

    it('should not mutate the original thread array', () => {
      const thread: Thread = [
        createMockChunk('c1', 0.8, 'Content 1'),
        createMockChunk('c2', 0.9, 'Content 2'),
      ];
      const threadCopy = [...thread];
      sortThreadChunks(thread);
      expect(thread).toEqual(threadCopy);
    });
  });

  describe('flattenThreadContent', () => {
    it('should join chunk content with double newlines', () => {
      const thread: Thread = [
        createMockChunk('c1', 0.9, 'First message.'),
        createMockChunk('c2', 0.8, 'Second message.'),
        createMockChunk('c3', 0.7, 'Third message.'),
      ];
      const expected = 'First message.\n\nSecond message.\n\nThird message.';
      expect(flattenThreadContent(thread)).toBe(expected);
    });

    it('should handle a thread with a single chunk', () => {
      const thread: Thread = [createMockChunk('c1', 0.9, 'Only message.')];
      expect(flattenThreadContent(thread)).toBe('Only message.');
    });

    it('should return an empty string for an empty thread', () => {
      const thread: Thread = [];
      expect(flattenThreadContent(thread)).toBe('');
    });
  });
});
