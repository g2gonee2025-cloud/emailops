import { renderHook, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { useDebounce } from './useDebounce';

describe('useDebounce', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should return the initial value immediately', () => {
    const { result } = renderHook(() => useDebounce('initial', 500));
    expect(result.current).toBe('initial');
  });

  it('should update the debounced value after the delay', () => {
    const { result, rerender } = renderHook(({ value, delay }) => useDebounce(value, delay), {
      initialProps: { value: 'initial', delay: 500 },
    });

    expect(result.current).toBe('initial');

    rerender({ value: 'updated', delay: 500 });

    // Value should not have updated yet
    expect(result.current).toBe('initial');

    // Fast-forward time
    act(() => {
      vi.advanceTimersByTime(500);
    });

    // Now the value should be updated
    expect(result.current).toBe('updated');
  });

  it('should reset the timeout if the value changes', () => {
    const { result, rerender } = renderHook(({ value, delay }) => useDebounce(value, delay), {
      initialProps: { value: 'initial', delay: 500 },
    });

    rerender({ value: 'updated', delay: 500 });

    act(() => {
      vi.advanceTimersByTime(250);
    });

    // The value should still be the initial one.
    expect(result.current).toBe('initial');

    rerender({ value: 'final', delay: 500 });
    expect(result.current).toBe('initial');

    // Fast-forward time past the first timer's trigger time
    act(() => {
      vi.advanceTimersByTime(250);
    });

    // The timer was reset, so the value should not have changed
    expect(result.current).toBe('initial');

    // Fast-forward time past the second timer's trigger time
    act(() => {
      vi.advanceTimersByTime(250);
    });

    // Now the value should be updated to the final value
    expect(result.current).toBe('final');
  });

  it('should handle rapid changes and only update to the last value', () => {
    const { result, rerender } = renderHook(({ value, delay }) => useDebounce(value, delay), {
      initialProps: { value: 'a', delay: 500 },
    });

    rerender({ value: 'b', delay: 500 });
    vi.advanceTimersByTime(100);
    rerender({ value: 'c', delay: 500 });
    vi.advanceTimersByTime(100);
    rerender({ value: 'd', delay: 500 });

    expect(result.current).toBe('a');

    // Fast-forward time to trigger the debounce for the last value 'd'
    act(() => {
      vi.advanceTimersByTime(500);
    });

    expect(result.current).toBe('d');
  });
});
