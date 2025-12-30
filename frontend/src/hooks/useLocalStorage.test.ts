import { renderHook, act } from '@testing-library/react';
import useLocalStorage from './useLocalStorage';

describe('useLocalStorage', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it('should return the initial value if localStorage is empty', () => {
    const { result } = renderHook(() => useLocalStorage('test-key', 'initial'));
    expect(result.current[0]).toBe('initial');
  });

  it('should return the stored value from localStorage', () => {
    window.localStorage.setItem('test-key', JSON.stringify('stored'));
    const { result } = renderHook(() => useLocalStorage('test-key', 'initial'));
    expect(result.current[0]).toBe('stored');
  });

  it('should update the value in localStorage when the state changes', () => {
    const { result } = renderHook(() => useLocalStorage('test-key', 'initial'));

    act(() => {
      result.current[1]('updated');
    });

    expect(result.current[0]).toBe('updated');
    expect(window.localStorage.getItem('test-key')).toBe(JSON.stringify('updated'));
  });

  it('should handle complex objects', () => {
    const initialObject = { a: 1, b: 'two' };
    const { result } = renderHook(() => useLocalStorage('test-key', initialObject));

    expect(result.current[0]).toEqual(initialObject);

    const updatedObject = { a: 2, b: 'three' };
    act(() => {
      result.current[1](updatedObject);
    });

    expect(result.current[0]).toEqual(updatedObject);
    expect(window.localStorage.getItem('test-key')).toBe(JSON.stringify(updatedObject));
  });

  it('should handle SSR by returning the initial value', () => {
    const originalLocalStorage = window.localStorage;
    Object.defineProperty(window, 'localStorage', {
      value: undefined,
      writable: true,
    });

    const { result } = renderHook(() => useLocalStorage('test-key', 'ssr-initial'));
    expect(result.current[0]).toBe('ssr-initial');

    Object.defineProperty(window, 'localStorage', {
      value: originalLocalStorage,
      writable: true,
    });
  });
});
