import { describe, it, expect } from 'vitest';
import { serializeSearchQuery, parseSearchQuery } from './searchUtils';
import type { SearchParams } from './searchUtils';

describe('searchUtils', () => {
  describe('serializeSearchQuery', () => {
    it('should serialize a full SearchParams object', () => {
      const params: SearchParams = {
        query: 'test query',
        k: 20,
        filters: { author: 'John Doe', status: 'open' },
      };

      expect(serializeSearchQuery(params)).toContain('query=test+query');
      expect(serializeSearchQuery(params)).toContain('k=20');
      expect(serializeSearchQuery(params)).toContain('filters=%7B%22author%22%3A%22John+Doe%22%2C%22status%22%3A%22open%22%7D');
    });

    it('should handle only a query', () => {
      const params = { query: 'hello' };
      expect(serializeSearchQuery(params)).toBe('?query=hello');
    });

    it('should handle only k', () => {
      const params = { k: 5 };
      expect(serializeSearchQuery(params)).toBe('?k=5');
    });

    it('should handle only filters', () => {
      const params = { filters: { category: 'urgent' } };
      expect(serializeSearchQuery(params)).toBe('?filters=%7B%22category%22%3A%22urgent%22%7D');
    });

    it('should return an empty string for an empty object', () => {
      const params = {};
      expect(serializeSearchQuery(params)).toBe('');
    });

    it('should not include filters if it is an empty object', () => {
      const params = { query: 'test', filters: {} };
      expect(serializeSearchQuery(params)).toBe('?query=test');
    });
  });

  describe('parseSearchQuery', () => {
    it('should parse a full query string', () => {
      const queryString = '?query=test+query&k=20&filters=%7B%22author%22%3A%22John+Doe%22%7D';
      const expected: SearchParams = {
        query: 'test query',
        k: 20,
        filters: { author: 'John Doe' },
      };
      expect(parseSearchQuery(queryString)).toEqual(expected);
    });

    it('should parse a query string with only a query', () => {
      const queryString = '?query=hello';
      const expected: SearchParams = {
        query: 'hello',
        k: 10, // default
        filters: {}, // default
      };
      expect(parseSearchQuery(queryString)).toEqual(expected);
    });

    it('should return default values for an empty query string', () => {
      const queryString = '';
      const expected: SearchParams = {
        query: '',
        k: 10,
        filters: {},
      };
      expect(parseSearchQuery(queryString)).toEqual(expected);
    });

    it('should handle invalid JSON in filters gracefully', () => {
      const queryString = '?query=test&filters=invalid-json';
      const result = parseSearchQuery(queryString);
      expect(result.query).toBe('test');
      expect(result.filters).toEqual({});
    });

    it('should handle non-object JSON in filters gracefully', () => {
      const queryString = '?query=test&filters=%5B%22array%22%5D'; // JSON for ["array"]
      const result = parseSearchQuery(queryString);
      expect(result.query).toBe('test');
      expect(result.filters).toEqual({});
    });

    it('should handle invalid k value gracefully', () => {
      const queryString = '?query=test&k=not-a-number';
      const result = parseSearchQuery(queryString);
      expect(result.query).toBe('test');
      expect(result.k).toBe(10); // default
    });

    it('should handle negative k value by using default', () => {
      const queryString = '?k=-5';
      const result = parseSearchQuery(queryString);
      expect(result.k).toBe(10); // default, because k must be positive
    });

    it('should handle float k value by coercing to int', () => {
        const queryString = '?k=15.7';
        const result = parseSearchQuery(queryString);
        expect(result.k).toBe(15);
      });
  });
});
