import { describe, it, expect } from 'vitest';
import { serializeSearchToURL, parseSearchFromURL } from './searchUtils';
import { type SearchParams } from '../schemas/search';

describe('searchUtils', () => {
  describe('serializeSearchToURL', () => {
    it('should serialize a full SearchParams object', () => {
      const params: SearchParams = {
        query: 'test query',
        k: 20,
        filters: { category: 'email', unread: true },
      };
      const result = serializeSearchToURL(params);
      expect(result.get('query')).toBe('test query');
      expect(result.get('k')).toBe('20');
      expect(result.get('filters')).toBe('{"category":"email","unread":true}');
    });

    it('should handle partial params', () => {
      const params: SearchParams = {
        query: 'hello',
        k: 10,
        filters: {},
      };
      const result = serializeSearchToURL(params);
      expect(result.get('query')).toBe('hello');
      expect(result.get('k')).toBe('10');
      expect(result.has('filters')).toBe(false);
    });

    it('should handle empty query', () => {
      const params: SearchParams = {
        query: '',
        k: 5,
        filters: { a: 1 },
      };
      const result = serializeSearchToURL(params);
      expect(result.has('query')).toBe(false);
      expect(result.get('k')).toBe('5');
      expect(result.get('filters')).toBe('{"a":1}');
    });
  });

  describe('parseSearchFromURL', () => {
    it('should parse a full URLSearchParams object', () => {
      const searchParams = new URLSearchParams();
      searchParams.set('query', 'test query');
      searchParams.set('k', '15');
      searchParams.set('filters', '{"category":"document"}');

      const result = parseSearchFromURL(searchParams);

      expect(result).toEqual({
        query: 'test query',
        k: 15,
        filters: { category: 'document' },
      });
    });

    it('should return defaults for an empty URLSearchParams', () => {
      const searchParams = new URLSearchParams();
      const result = parseSearchFromURL(searchParams);
      expect(result).toEqual({
        query: '',
        k: 10,
        filters: {},
      });
    });

    it('should handle malformed k value', () => {
      const searchParams = new URLSearchParams();
      searchParams.set('query', 'test');
      searchParams.set('k', 'abc');
      const result = parseSearchFromURL(searchParams);
      expect(result.k).toBe(10); // default
      expect(result.query).toBe('test');
    });

    it('should handle malformed filters JSON', () => {
      const searchParams = new URLSearchParams();
      searchParams.set('filters', '{"cat:"doc"'); // invalid JSON
      const result = parseSearchFromURL(searchParams);
      expect(result.filters).toEqual({}); // default
    });

    it('should handle non-object filters', () => {
      const searchParams = new URLSearchParams();
      searchParams.set('filters', '123');
      const result = parseSearchFromURL(searchParams);
      expect(result.filters).toEqual({}); // default
    });

    it('should ignore extra parameters', () => {
      const searchParams = new URLSearchParams();
      searchParams.set('query', 'test');
      searchParams.set('utm_source', 'google');
      const result = parseSearchFromURL(searchParams);
      expect(result).toEqual({
        query: 'test',
        k: 10,
        filters: {},
      });
    });
  });
});
