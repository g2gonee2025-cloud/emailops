/**
 * @vitest-environment jsdom
 */
import { describe, it, expect } from 'vitest';
import { v4 as uuidv4 } from 'uuid';
import {
  searchResultSchema,
  searchResponseSchema,
  isSearchResult,
  isSearchResponse,
  type SearchResult,
} from './search';

// =======================================================================
// Test Data
// =======================================================================

const createMockSearchResult = (overrides: Partial<SearchResult> = {}): SearchResult => ({
  chunk_id: uuidv4(),
  conversation_id: uuidv4(),
  thread_id: uuidv4(),
  content: 'This is a test search result.',
  score: 0.85,
  metadata: { source: 'email' },
  ...overrides,
});

// =======================================================================
// Tests for searchResultSchema
// =======================================================================

describe('searchResultSchema', () => {
  it('should validate a correct search result object', () => {
    const validResult = createMockSearchResult();
    const validation = searchResultSchema.safeParse(validResult);
    expect(validation.success).toBe(true);
  });

  it('should allow optional thread_id and metadata', () => {
    const result = createMockSearchResult();
    delete result.thread_id;
    delete result.metadata;

    const validation = searchResultSchema.safeParse(result);
    expect(validation.success).toBe(true);
  });

  it('should fail validation for an invalid chunk_id', () => {
    const invalidResult = createMockSearchResult({ chunk_id: 'not-a-uuid' });
    const validation = searchResultSchema.safeParse(invalidResult);
    expect(validation.success).toBe(false);
    // @ts-expect-error - checking for error
    expect(validation.error.errors[0].message).toContain('Invalid chunk ID');
  });

  it('should fail validation for a negative score', () => {
    const invalidResult = createMockSearchResult({ score: -0.5 });
    const validation = searchResultSchema.safeParse(invalidResult);
    expect(validation.success).toBe(false);
    // @ts-expect-error - checking for error
    expect(validation.error.errors[0].message).toContain('Score must be non-negative');
  });

  it('should fail if required fields are missing', () => {
    const incompleteResult = {
      chunk_id: uuidv4(),
      content: 'Incomplete data',
    };
    const validation = searchResultSchema.safeParse(incompleteResult);
    expect(validation.success).toBe(false);
    // @ts-expect-error - checking for error
    expect(validation.error.issues.some(issue => issue.path.includes('conversation_id'))).toBe(true);
    // @ts-expect-error - checking for error
    expect(validation.error.issues.some(issue => issue.path.includes('score'))).toBe(true);
  });
});


// =======================================================================
// Tests for searchResponseSchema
// =======================================================================

describe('searchResponseSchema', () => {
  it('should validate a correct search response object', () => {
    const validResponse = {
      results: [createMockSearchResult(), createMockSearchResult()],
      total_count: 2,
      query_time_ms: 120.5,
      correlation_id: `corr-${uuidv4()}`,
    };
    const validation = searchResponseSchema.safeParse(validResponse);
    expect(validation.success).toBe(true);
  });

  it('should allow an empty results array', () => {
    const emptyResponse = {
      results: [],
      total_count: 0,
      query_time_ms: 50,
    };
    const validation = searchResponseSchema.safeParse(emptyResponse);
    expect(validation.success).toBe(true);
  });

  it('should fail validation if total_count is negative', () => {
    const invalidResponse = {
      results: [],
      total_count: -1,
      query_time_ms: 50,
    };
    const validation = searchResponseSchema.safeParse(invalidResponse);
    expect(validation.success).toBe(false);
    // @ts-expect-error - checking for error
    expect(validation.error.errors[0].message).toContain('Number must be greater than or equal to 0');
  });

  it('should fail if results array contains invalid objects', () => {
    const invalidResponse = {
      results: [createMockSearchResult(), { chunk_id: 'invalid' }],
      total_count: 2,
      query_time_ms: 100,
    };
    const validation = searchResponseSchema.safeParse(invalidResponse);
    expect(validation.success).toBe(false);
  });
});

// =======================================================================
// Tests for Helper Validators
// =======================================================================

describe('Helper Validators', () => {
  it('isSearchResult should return true for valid data', () => {
    const validData = createMockSearchResult();
    expect(isSearchResult(validData)).toBe(true);
  });

  it('isSearchResult should return false for invalid data', () => {
    const invalidData = { content: 'test', score: -1 };
    expect(isSearchResult(invalidData)).toBe(false);
  });

  it('isSearchResponse should return true for valid data', () => {
    const validData = {
      results: [createMockSearchResult()],
      total_count: 1,
      query_time_ms: 100,
    };
    expect(isSearchResponse(validData)).toBe(true);
  });

  it('isSearchResponse should return false for invalid data', () => {
    const invalidData = { results: [], total_count: -5 };
    expect(isSearchResponse(invalidData)).toBe(false);
  });
});
