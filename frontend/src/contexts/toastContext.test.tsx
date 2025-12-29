/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, vi } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import { ToastProvider, useToast } from './toastContext';
import { ApiError } from '../lib/api';

// A simple component to trigger toasts
function TestComponent() {
  const { addToast } = useToast();
  return (
    <div>
      <button onClick={() => addToast({ message: 'Test success', type: 'success' })}>Add Success Toast</button>
      <button onClick={() => addToast({ message: 'Test error', type: 'error', details: 'Error details' })}>
        Add Error Toast
      </button>
    </div>
  );
}

describe('ToastProvider', () => {
  it('should add and remove toasts', () => {
    render(
      <ToastProvider>
        <TestComponent />
      </ToastProvider>,
    );

    act(() => {
      screen.getByText('Add Success Toast').click();
    });

    expect(screen.getByText('Test success')).toBeInTheDocument();
  });

  it('should display toast with details', () => {
    render(
      <ToastProvider>
        <TestComponent />
      </ToastProvider>,
    );

    act(() => {
      screen.getByText('Add Error Toast').click();
    });

    expect(screen.getByText('Test error')).toBeInTheDocument();
    expect(screen.getByText('Error details')).toBeInTheDocument();
  });

  it('should handle global api:error event', () => {
    render(
      <ToastProvider>
        <div />
      </ToastProvider>,
    );

    const error = new ApiError(500, 'Internal Server Error', 'Something broke');

    // Mock console.error to avoid polluting test output
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    act(() => {
      window.dispatchEvent(new CustomEvent('api:error', { detail: error }));
    });

    expect(screen.getByText(error.detail!)).toBeInTheDocument();

    // In dev, it should also show status and statusText
    const isDev = import.meta.env.DEV;
    if (isDev) {
      expect(screen.getByText(`[${error.status}] ${error.statusText}`)).toBeInTheDocument();
    }

    expect(consoleErrorSpy).toHaveBeenCalledWith('Global API Error:', error);

    consoleErrorSpy.mockRestore();
  });
});
