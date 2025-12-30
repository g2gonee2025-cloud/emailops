/**
 * @file Utilities for sorting and processing thread chunks.
 * @description These are pure functions that operate on thread data structures.
 */

import type { SearchResult } from './api';

/**
 * A Thread is represented as an array of SearchResult chunks.
 */
export type Thread = SearchResult[];

/**
 * Sorts the chunks within a thread. The primary sort key is relevance score (descending),
 * with a secondary sort on the chunk_id (ascending) to ensure stability.
 *
 * @param thread - The array of SearchResult chunks to sort.
 * @returns A new array of sorted SearchResult chunks.
 */
export const sortThreadChunks = (thread: Thread): Thread => {
  return [...thread].sort((a, b) => {
    if (a.score !== b.score) {
      return b.score - a.score;
    }
    return a.chunk_id.localeCompare(b.chunk_id);
  });
};

/**
 * Flattens the content of all chunks in a thread into a single string,
 * with each chunk's content separated by a double newline.
 *
 * @param thread - The array of SearchResult chunks to flatten.
 * @returns A single string containing the concatenated content.
 */
export const flattenThreadContent = (thread: Thread): string => {
  return thread.map(chunk => chunk.content).join('\n\n');
};
