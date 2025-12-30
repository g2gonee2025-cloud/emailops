import { describe, it, expect } from 'vitest';
import { SearchResultSchema, SearchResponseSchema } from './search';

describe('Search Schemas', () => {
  describe('SearchResultSchema', () => {
    it('should validate a correct search result', () => {
      const validResult = {
        chunk_id: 'abc-123',
        conversation_id: 'conv-456',
        thread_id: 'thread-789',
        content: 'This is a valid piece of content.',
        score: 0.88,
        metadata: { source: 'email' },
      };
      const result = SearchResultSchema.safeParse(validResult);
      expect(result.success, 'Validation should succeed with a valid result').toBe(true);
    });

    it('should pass validation even with optional fields missing', () => {
      const validResultMinimal = {
        chunk_id: 'abc-123',
        conversation_id: 'conv-456',
        content: 'This is valid content.',
        score: 0.5,
      };
      const result = SearchResultSchema.safeParse(validResultMinimal);
      expect(result.success, 'Validation should succeed with minimal required fields').toBe(true);
    });

    it('should invalidate a search result with an empty chunk_id', () => {
      const invalidResult = {
        chunk_id: '',
        conversation_id: 'conv-456',
        content: 'Content here',
        score: 0.8,
      };
      const result = SearchResultSchema.safeParse(invalidResult);
      expect(result.success, 'Validation should fail with empty chunk_id').toBe(false);
    });

    it('should invalidate a search result with a negative score', () => {
      const invalidResult = {
        chunk_id: 'abc-123',
        conversation_id: 'conv-456',
        content: 'Content here',
        score: -0.5,
      };
      const result = SearchResultSchema.safeParse(invalidResult);
      expect(result.success, 'Validation should fail with a negative score').toBe(false);
    });
  });

  describe('SearchResponseSchema', () => {
    it('should validate a correct search response', () => {
      const validResponse = {
        results: [
          {
            chunk_id: 'abc-123',
            conversation_id: 'conv-456',
            content: 'This is a snippet.',
            score: 0.9,
          },
        ],
        total_count: 1,
        query_time_ms: 78,
        correlation_id: 'corr-id-123',
      };
      const result = SearchResponseSchema.safeParse(validResponse);
      expect(result.success, 'Validation should succeed for a valid response').toBe(true);
    });

    it('should validate a response with empty results', () => {
      const validEmptyResponse = {
        results: [],
        total_count: 0,
        query_time_ms: 10,
      };
      const result = SearchResponseSchema.safeParse(validEmptyResponse);
      expect(result.success, 'Validation should succeed for an empty but valid response').toBe(true);
    });

    it('should invalidate a response with a negative total_count', () => {
      const invalidResponse = {
        results: [],
        total_count: -1,
        query_time_ms: 10,
      };
      const result = SearchResponseSchema.safeParse(invalidResponse);
      expect(result.success, 'Validation should fail for a negative total_count').toBe(false);
    });

    it('should invalidate a response with an incorrect results array structure', () => {
      const invalidResponse = {
        results: [{ id: 'wrong-field' }], // Incorrect object structure
        total_count: 1,
        query_time_ms: 15,
      };
      const result = SearchResponseSchema.safeParse(invalidResponse);
      expect(result.success, 'Validation should fail for malformed items in results array').toBe(false);
    });
  });
});
