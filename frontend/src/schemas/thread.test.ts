/** @vitest-environment jsdom */
import { describe, it, expect } from 'vitest';
import { SearchResultSchema, ThreadSummarySchema } from './thread';

describe('Thread Schemas', () => {
  describe('SearchResultSchema', () => {
    it('should validate a correct search result object', () => {
      const validSearchResult = {
        chunk_id: 'a1b2c3d4-e5f6-7890-1234-567890abcdef',
        conversation_id: 'conv-123',
        content: 'This is a test chunk.',
        score: 0.95,
      };
      const result = SearchResultSchema.safeParse(validSearchResult);
      expect(result.success).toBe(true);
    });

    it('should fail validation for an invalid search result object', () => {
      const invalidSearchResult = {
        chunk_id: 'not-a-uuid',
        content: 123, // should be a string
        score: 'high', // should be a number
      };
      const result = SearchResultSchema.safeParse(invalidSearchResult);
      expect(result.success).toBe(false);
    });
  });

  describe('ThreadSummarySchema', () => {
    it('should validate a correct thread summary object', () => {
      const validThreadSummary = {
        summary: 'This is a test summary.',
        key_points: ['point 1', 'point 2'],
      };
      const result = ThreadSummarySchema.safeParse(validThreadSummary);
      expect(result.success).toBe(true);
    });

    it('should fail validation for an invalid thread summary object', () => {
      const invalidThreadSummary = {
        summary: 123, // should be a string
        key_points: 'point 1', // should be an array of strings
      };
      const result = ThreadSummarySchema.safeParse(invalidThreadSummary);
      expect(result.success).toBe(false);
    });
  });
});
