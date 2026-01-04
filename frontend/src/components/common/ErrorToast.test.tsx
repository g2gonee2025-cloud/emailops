/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, act, fireEvent, waitFor } from '@testing-library/react';
import {
  ErrorToastContainer,
  ErrorToastItem,
  useErrorToast,
  type ErrorToastData,
} from './ErrorToast';
import { ApiError } from '../../lib/api';

function TestErrorToastHook({ onAuthRedirect }: { onAuthRedirect?: () => void }) {
  const { toasts, addErrorToast, removeToast } = useErrorToast({ onAuthRedirect });
  return (
    <div>
      <button
        onClick={() => {
          const error = new ApiError('Test error', 500, {
            error: {
              error_code: 'INTERNAL_ERROR',
              message: 'Something went wrong',
              correlation_id: 'test-correlation-123',
              retryable: true,
            },
          });
          addErrorToast(error);
        }}
      >
        Add Error
      </button>
      <button
        onClick={() => {
          const error = new ApiError('Auth error', 401, {
            error: {
              error_code: 'AUTH_REQUIRED',
              message: 'Authentication required',
              correlation_id: 'auth-123',
              retryable: false,
            },
          });
          addErrorToast(error);
        }}
      >
        Add Auth Error
      </button>
      <button
        onClick={() => {
          const error = new ApiError('Validation error', 400, {
            error: {
              error_code: 'VALIDATION_ERROR',
              message: 'Invalid input',
              correlation_id: 'val-123',
              retryable: false,
            },
          });
          addErrorToast(error);
        }}
      >
        Add Validation Error
      </button>
      <ErrorToastContainer toasts={toasts} removeToast={removeToast} />
    </div>
  );
}

describe('ErrorToast Components', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('ErrorToastContainer', () => {
    it('should render empty container when no toasts', () => {
      render(<ErrorToastContainer toasts={[]} removeToast={vi.fn()} />);
      const container = screen.getByTestId('error-toast-container');
      expect(container).toBeInTheDocument();
      expect(container.children).toHaveLength(0);
    });

    it('should render multiple toasts', () => {
      const toasts: ErrorToastData[] = [
        {
          id: '1',
          message: 'Error 1',
          severity: 'error',
          correlationId: 'corr-1',
          retryable: false,
        },
        {
          id: '2',
          message: 'Error 2',
          severity: 'warning',
          correlationId: 'corr-2',
          retryable: true,
          actionLabel: 'Retry',
        },
      ];

      render(<ErrorToastContainer toasts={toasts} removeToast={vi.fn()} />);

      expect(screen.getByText('Error 1')).toBeInTheDocument();
      expect(screen.getByText('Error 2')).toBeInTheDocument();
    });
  });

  describe('ErrorToastItem', () => {
    it('should render toast with message and correlation ID', () => {
      const toast: ErrorToastData = {
        id: 'test-1',
        message: 'Test error message',
        severity: 'error',
        correlationId: 'abc-123-def',
        retryable: false,
      };

      render(<ErrorToastItem toast={toast} onRemove={vi.fn()} />);

      expect(screen.getByText('Test error message')).toBeInTheDocument();
      expect(screen.getByText('abc-123-def')).toBeInTheDocument();
    });

    it('should render retry button when retryable with actionLabel', () => {
      const toast: ErrorToastData = {
        id: 'test-1',
        message: 'Retryable error',
        severity: 'warning',
        correlationId: 'xyz-789',
        retryable: true,
        actionLabel: 'Retry',
      };

      render(<ErrorToastItem toast={toast} onRemove={vi.fn()} />);

      expect(screen.getByTestId('retry-button')).toBeInTheDocument();
      expect(screen.getByText('Retry')).toBeInTheDocument();
    });

    it('should not render retry button when not retryable', () => {
      const toast: ErrorToastData = {
        id: 'test-1',
        message: 'Non-retryable error',
        severity: 'error',
        correlationId: 'xyz-789',
        retryable: false,
      };

      render(<ErrorToastItem toast={toast} onRemove={vi.fn()} />);

      expect(screen.queryByTestId('retry-button')).not.toBeInTheDocument();
    });

    it('should call onRemove when dismiss button is clicked', () => {
      const onRemove = vi.fn();
      const toast: ErrorToastData = {
        id: 'test-1',
        message: 'Test error',
        severity: 'error',
        correlationId: null,
        retryable: false,
      };

      render(<ErrorToastItem toast={toast} onRemove={onRemove} />);

      fireEvent.click(screen.getByLabelText('Dismiss notification'));
      expect(onRemove).toHaveBeenCalledWith('test-1');
    });

    it('should call onRetry and onRemove when retry button is clicked', () => {
      const onRemove = vi.fn();
      const onRetry = vi.fn();
      const toast: ErrorToastData = {
        id: 'test-1',
        message: 'Retryable error',
        severity: 'warning',
        correlationId: null,
        retryable: true,
        actionLabel: 'Retry',
        onRetry,
      };

      render(<ErrorToastItem toast={toast} onRemove={onRemove} />);

      fireEvent.click(screen.getByTestId('retry-button'));
      expect(onRetry).toHaveBeenCalled();
      expect(onRemove).toHaveBeenCalledWith('test-1');
    });

    it('should copy correlation ID to clipboard when copy button is clicked', async () => {
      const writeText = vi.fn().mockResolvedValue(undefined);
      Object.assign(navigator, {
        clipboard: { writeText },
      });

      const toast: ErrorToastData = {
        id: 'test-1',
        message: 'Test error',
        severity: 'error',
        correlationId: 'copy-me-123',
        retryable: false,
      };

      render(<ErrorToastItem toast={toast} onRemove={vi.fn()} />);

      fireEvent.click(screen.getByTestId('copy-correlation-id'));

      await waitFor(() => {
        expect(writeText).toHaveBeenCalledWith('copy-me-123');
      });
    });

    it('should apply correct severity styling for error', () => {
      const toast: ErrorToastData = {
        id: 'test-1',
        message: 'Error toast',
        severity: 'error',
        correlationId: null,
        retryable: false,
      };

      render(<ErrorToastItem toast={toast} onRemove={vi.fn()} />);

      const toastElement = screen.getByTestId('error-toast');
      expect(toastElement).toHaveClass('border-red-500/30');
      expect(toastElement).toHaveClass('bg-red-500/10');
    });

    it('should apply correct severity styling for warning', () => {
      const toast: ErrorToastData = {
        id: 'test-1',
        message: 'Warning toast',
        severity: 'warning',
        correlationId: null,
        retryable: false,
      };

      render(<ErrorToastItem toast={toast} onRemove={vi.fn()} />);

      const toastElement = screen.getByTestId('error-toast');
      expect(toastElement).toHaveClass('border-yellow-500/30');
      expect(toastElement).toHaveClass('bg-yellow-500/10');
    });

    it('should apply correct severity styling for critical', () => {
      const toast: ErrorToastData = {
        id: 'test-1',
        message: 'Critical toast',
        severity: 'critical',
        correlationId: null,
        retryable: false,
      };

      render(<ErrorToastItem toast={toast} onRemove={vi.fn()} />);

      const toastElement = screen.getByTestId('error-toast');
      expect(toastElement).toHaveClass('border-red-600/40');
      expect(toastElement).toHaveClass('bg-red-600/15');
    });
  });

  describe('useErrorToast hook', () => {
    it('should add toast for INTERNAL_ERROR', () => {
      render(<TestErrorToastHook />);

      act(() => {
        screen.getByText('Add Error').click();
      });

      expect(screen.getByText('Something went wrong')).toBeInTheDocument();
      expect(screen.getByText('test-correlation-123')).toBeInTheDocument();
    });

    it('should call onAuthRedirect for AUTH_REQUIRED error', () => {
      const onAuthRedirect = vi.fn();
      render(<TestErrorToastHook onAuthRedirect={onAuthRedirect} />);

      act(() => {
        screen.getByText('Add Auth Error').click();
      });

      expect(onAuthRedirect).toHaveBeenCalled();
      expect(screen.queryByText('Authentication required')).not.toBeInTheDocument();
    });

    it('should not add toast for VALIDATION_ERROR (inline pattern)', () => {
      render(<TestErrorToastHook />);

      act(() => {
        screen.getByText('Add Validation Error').click();
      });

      expect(screen.queryByText('Invalid input')).not.toBeInTheDocument();
    });

    it('should handle api:error event', () => {
      render(<TestErrorToastHook />);

      const error = new ApiError('Server error', 500, {
        error: {
          error_code: 'INTERNAL_ERROR',
          message: 'Server crashed',
          correlation_id: 'event-123',
          retryable: true,
        },
      });

      act(() => {
        window.dispatchEvent(new CustomEvent('api:error', { detail: error }));
      });

      expect(screen.getByText('Server crashed')).toBeInTheDocument();
      expect(screen.getByText('event-123')).toBeInTheDocument();
    });
  });
});
