/**
 * Thread Utilities
 *
 * This module provides pure functions for sorting, flattening, and otherwise
 * manipulating thread chunks and messages. Its purpose is to centralize
 * common thread operations and ensure consistent handling of thread data
 * structures throughout the frontend application.
 *
 * As pure functions, these utilities are deterministic and have no side effects,
 * making them easy to test and reason about.
 */

import { ChatMessage, SearchResult } from './api';

/**
 * A generic type representing any object that has a timestamp.
 * The timestamp can be either a string or a number.
 */
type Timestamped =
  | { metadata?: { timestamp?: string | number } }
  | { timestamp?: string | number };

/**
 * Sorts an array of objects by a timestamp property.
 *
 * The function is generic and can handle any object that has a `timestamp`
 * property or a `metadata.timestamp` property. The timestamp can be a string
 * in ISO 8601 format or a number (Unix epoch).
 *
 * @param items The array of items to sort.
 * @param order The sort order, either 'asc' for ascending or 'desc' for descending.
 * @returns A new array with the items sorted by timestamp.
 */
export const sortByTimestamp = <T extends Timestamped>(
  items: T[],
  order: 'asc' | 'desc' = 'asc',
): T[] => {
  return [...items].sort((a, b) => {
    const tsA = 'metadata' in a ? a.metadata?.timestamp : a.timestamp;
    const tsB = 'metadata' in b ? b.metadata?.timestamp : b.timestamp;

    if (tsA === undefined || tsB === undefined) {
      return 0;
    }

    const dateA = new Date(tsA).getTime();
    const dateB = new Date(tsB).getTime();

    if (isNaN(dateA) || isNaN(dateB)) {
      return 0;
    }

    return order === 'asc' ? dateA - dateB : dateB - dateA;
  });
};

/**
 * Flattens an array of threads into a single array of items.
 *
 * This function is useful for combining search results or messages from
 * multiple threads into a single list for display or processing.
 *
 * @param threads An array of threads, where each thread is an array of items.
 * @returns A new array containing all the items from all threads.
 */
export const flattenThreads = <T>(threads: T[][]): T[] => {
  return threads.flat();
};
