/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, vi } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import { ToastProvider, useToast } from './toastContext';
import { ApiError } from '../lib/api';
import { logger } from '../lib/logger';

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

  it('should handle global api:error event and show a toast', async () => {
    render(
      <ToastProvider>
        <div />
      </ToastProvider>,
    );

    const errorDetails = { detail: 'Something broke on the server.' };
    const error = new ApiError('Internal Server Error', 500, errorDetails);

    // Spy on the logger to ensure it's called
    const loggerSpy = vi.spyOn(logger, 'error').mockImplementation(() => {});

    // Dispatch the custom event
    act(() => {
      window.dispatchEvent(new CustomEvent('api:error', { detail: error }));
    });

    // Verify the toast is displayed with the correct content
    expect(await screen.findByText(errorDetails.detail)).toBeInTheDocument();

    // In dev, it should also show status and message
    const isDev = import.meta.env.DEV;
    if (isDev) {
      expect(await screen.findByText(`[${error.status}] ${error.message}`)).toBeInTheDocument();
    }

    // Verify the logger was called
    expect(loggerSpy).toHaveBeenCalledWith('Global API Error Event', {
      name: 'ApiError',
      message: error.message,
      status: error.status,
      details: errorDetails,
    });

    loggerSpy.mockRestore();
  });
});
