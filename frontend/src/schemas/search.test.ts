import { describe, it, expect } from 'vitest';
import {
  SearchResultSchema,
  SearchFiltersSchema,
  SearchRequestSchema,
  SearchResponseSchema,
} from './search';

describe('Search Schemas', () => {
  describe('SearchResultSchema', () => {
    it('should validate a correct search result', () => {
      const validResult = {
        id: '123',
        title: 'Test Title',
        snippet: 'This is a snippet.',
        url: 'https://example.com',
      };
      const result = SearchResultSchema.safeParse(validResult);
      expect(result.success).toBe(true);
    });

    it('should invalidate a search result with an empty id', () => {
      const invalidResult = {
        id: '',
        title: 'Test Title',
        snippet: 'This is a snippet.',
        url: 'https://example.com',
      };
      const result = SearchResultSchema.safeParse(invalidResult);
      expect(result.success).toBe(false);
    });

    it('should invalidate a search result with an invalid URL', () => {
      const invalidResult = {
        id: '123',
        title: 'Test Title',
        snippet: 'This is a snippet.',
        url: 'not-a-url',
      };
      const result = SearchResultSchema.safeParse(invalidResult);
      expect(result.success).toBe(false);
    });
  });

  describe('SearchFiltersSchema', () => {
    it('should validate correct filters', () => {
      const validFilters = {
        filetype: 'pdf',
        author: 'John Doe',
        dateRange: { startDate: '2023-01-01', endDate: '2023-12-31' },
      };
      const result = SearchFiltersSchema.safeParse(validFilters);
      expect(result.success).toBe(true);
    });

    it('should validate with optional fields missing', () => {
      const partialFilters = { filetype: 'pdf' };
      const result = SearchFiltersSchema.safeParse(partialFilters);
      expect(result.success).toBe(true);
    });

    it('should validate an empty filters object', () => {
      const emptyFilters = {};
      const result = SearchFiltersSchema.safeParse(emptyFilters);
      expect(result.success).toBe(true);
    });
  });

  describe('SearchRequestSchema', () => {
    it('should validate a correct search request', () => {
      const validRequest = {
        query: 'test query',
        filters: { filetype: 'pdf' },
      };
      const result = SearchRequestSchema.safeParse(validRequest);
      expect(result.success).toBe(true);
    });

    it('should invalidate a request with an empty query', () => {
      const invalidRequest = { query: '' };
      const result = SearchRequestSchema.safeParse(invalidRequest);
      expect(result.success).toBe(false);
    });
  });

  describe('SearchResponseSchema', () => {
    it('should validate a correct search response', () => {
      const validResponse = {
        results: [
          {
            id: '123',
            title: 'Test Title',
            snippet: 'This is a snippet.',
            url: 'https://example.com',
          },
        ],
        total: 1,
        page: 1,
        pageSize: 10,
      };
      const result = SearchResponseSchema.safeParse(validResponse);
      expect(result.success).toBe(true);
    });

    it('should invalidate a response with a negative total', () => {
      const invalidResponse = {
        results: [],
        total: -1,
        page: 1,
        pageSize: 10,
      };
      const result = SearchResponseSchema.safeParse(invalidResponse);
      expect(result.success).toBe(false);
    });
  });
});
