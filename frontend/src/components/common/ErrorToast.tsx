/**
 * ErrorToast Component
 *
 * A specialized toast component for displaying API errors with:
 * - Correlation ID with copy button
 * - Retry button for retryable errors
 * - Severity-based styling
 * - Listens to `api:error` CustomEvent
 */
/* eslint-disable react-refresh/only-export-components */
import { useEffect, useState, useCallback } from 'react';
import { cn } from '../../lib/utils';
import { AlertCircle, AlertTriangle, Info, X, Copy, Check, RefreshCw } from 'lucide-react';
import { Button } from '../ui/Button';
import { type ApiError } from '../../lib/api';
import {
  getErrorConfig,
  isToastError,
  shouldRedirectToLogin,
  type ErrorSeverity,
} from '../../lib/errors';

export interface ErrorToastData {
  id: string;
  message: string;
  severity: ErrorSeverity;
  correlationId: string | null;
  retryable: boolean;
  actionLabel?: string;
  autoHide?: number;
  onRetry?: () => void;
}

interface ErrorToastItemProps {
  toast: ErrorToastData;
  onRemove: (id: string) => void;
}

function ErrorToastItem({ toast, onRemove }: ErrorToastItemProps) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (toast.autoHide && toast.autoHide > 0) {
      const timer = setTimeout(() => onRemove(toast.id), toast.autoHide);
      return () => clearTimeout(timer);
    }
  }, [toast.id, toast.autoHide, onRemove]);

  const handleCopyCorrelationId = useCallback(async () => {
    if (!toast.correlationId) return;
    try {
      await navigator.clipboard.writeText(toast.correlationId);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      console.error('Failed to copy correlation ID');
    }
  }, [toast.correlationId]);

  const handleRetry = useCallback(() => {
    if (toast.onRetry) {
      toast.onRetry();
    }
    onRemove(toast.id);
  }, [toast, onRemove]);

  const icons: Record<ErrorSeverity, React.ReactNode> = {
    info: <Info className="w-5 h-5 text-blue-400" />,
    warning: <AlertTriangle className="w-5 h-5 text-yellow-400" />,
    error: <AlertCircle className="w-5 h-5 text-red-400" />,
    critical: <AlertCircle className="w-5 h-5 text-red-500" />,
  };

  const styles: Record<ErrorSeverity, string> = {
    info: 'border-blue-500/30 bg-blue-500/10',
    warning: 'border-yellow-500/30 bg-yellow-500/10',
    error: 'border-red-500/30 bg-red-500/10',
    critical: 'border-red-600/40 bg-red-600/15',
  };

  return (
    <div
      className={cn(
        'flex flex-col gap-2 px-4 py-3 rounded-xl border backdrop-blur-xl shadow-lg',
        'animate-slide-up min-w-[320px] max-w-[450px]',
        styles[toast.severity],
      )}
      role="alert"
      aria-live="assertive"
      aria-atomic="true"
      data-testid="error-toast"
    >
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0 pt-0.5">{icons[toast.severity]}</div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-white/90">{toast.message}</p>
        </div>
        <Button
          type="button"
          onClick={() => onRemove(toast.id)}
          variant="ghost"
          size="icon"
          className="h-7 w-7 flex-shrink-0"
          aria-label="Dismiss notification"
        >
          <X className="w-4 h-4 text-white/50" />
        </Button>
      </div>

      <div className="flex items-center justify-between gap-2 mt-1">
        {toast.correlationId && (
          <button
            type="button"
            onClick={handleCopyCorrelationId}
            className="flex items-center gap-1.5 text-xs text-white/50 hover:text-white/70 transition-colors"
            title="Copy correlation ID"
            data-testid="copy-correlation-id"
          >
            {copied ? (
              <Check className="w-3 h-3 text-green-400" />
            ) : (
              <Copy className="w-3 h-3" />
            )}
            <span className="font-mono truncate max-w-[180px]">
              {toast.correlationId}
            </span>
          </button>
        )}

        {toast.retryable && toast.actionLabel && (
          <Button
            type="button"
            onClick={handleRetry}
            variant="ghost"
            size="sm"
            className="h-7 px-2 text-xs ml-auto"
            data-testid="retry-button"
          >
            <RefreshCw className="w-3 h-3 mr-1" />
            {toast.actionLabel}
          </Button>
        )}
      </div>
    </div>
  );
}

interface ErrorToastContainerProps {
  toasts: ErrorToastData[];
  removeToast: (id: string) => void;
}

export function ErrorToastContainer({ toasts, removeToast }: ErrorToastContainerProps) {
  return (
    <div
      className="fixed bottom-6 right-6 z-[100] flex flex-col gap-3"
      role="region"
      aria-label="Error notifications"
      data-testid="error-toast-container"
    >
      {toasts.map(toast => (
        <ErrorToastItem key={toast.id} toast={toast} onRemove={removeToast} />
      ))}
    </div>
  );
}

export interface UseErrorToastOptions {
  onAuthRedirect?: () => void;
}

export function useErrorToast(options: UseErrorToastOptions = {}) {
  const [toasts, setToasts] = useState<ErrorToastData[]>([]);

  const addErrorToast = useCallback(
    (error: ApiError, onRetry?: () => void) => {
      const config = getErrorConfig(error.errorCode);

      if (shouldRedirectToLogin(error.errorCode)) {
        options.onAuthRedirect?.();
        return;
      }

      if (!isToastError(error.errorCode)) {
        return;
      }

      const id = crypto.randomUUID();
      const newToast: ErrorToastData = {
        id,
        message: error.userMessage,
        severity: config.severity,
        correlationId: error.correlationId,
        retryable: config.retryable || error.retryable,
        actionLabel: config.actionLabel,
        autoHide: config.autoHide,
        onRetry,
      };

      setToasts(prev => [...prev, newToast]);
    },
    [options],
  );

  const removeToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  useEffect(() => {
    const handleApiError = (event: Event) => {
      const customEvent = event as CustomEvent<ApiError>;
      const error = customEvent.detail;
      addErrorToast(error);
    };

    globalThis.addEventListener('api:error', handleApiError);
    return () => {
      globalThis.removeEventListener('api:error', handleApiError);
    };
  }, [addErrorToast]);

  return {
    toasts,
    addErrorToast,
    removeToast,
  };
}

export { ErrorToastItem };
