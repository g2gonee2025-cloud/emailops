/**
 * Toast UI Components
 *
 * This file contains the visual components for displaying toasts. The logic and
 * state management are handled by the ToastProvider in `toastContext.tsx`.
 */
import { useEffect } from 'react';
import { cn } from '../../lib/utils';
import { CheckCircle, AlertCircle, Info, X, AlertTriangle } from 'lucide-react';
import { type Toast } from '../../contexts/toastContext';
import { Button } from './Button';

/**
 * A container that renders all active toasts.
 * This is an internal component used by the ToastProvider.
 */
export function ToastContainer({
  toasts,
  removeToast,
}: {
  toasts: Toast[];
  removeToast: (id: string) => void;
}) {
  return (
    <div
      className="fixed bottom-6 right-6 z-[100] flex flex-col gap-3"
      role="region"
      aria-label="Notifications"
    >
      {toasts.map(toast => (
        <ToastItem key={toast.id} toast={toast} onRemove={removeToast} />
      ))}
    </div>
  );
}

/**
 * An individual toast item.
 * This is an internal component used by the ToastContainer.
 */
function ToastItem({ toast, onRemove }: { toast: Toast; onRemove: (id: string) => void }) {
  // Auto-dismiss timer
  useEffect(() => {
    if (toast.duration && toast.duration > 0) {
      const timer = setTimeout(() => onRemove(toast.id), toast.duration);
      return () => clearTimeout(timer);
    }
  }, [toast.id, toast.duration, onRemove]);

  // Icon mapping
  const icons = {
    success: <CheckCircle className="w-5 h-5 text-green-400" />,
    error: <AlertCircle className="w-5 h-5 text-red-400" />,
    warning: <AlertTriangle className="w-5 h-5 text-yellow-400" />,
    info: <Info className="w-5 h-5 text-blue-400" />,
  };

  // Style mapping
  const styles = {
    success: 'border-green-500/30 bg-green-500/10',
    error: 'border-red-500/30 bg-red-500/10',
    warning: 'border-yellow-500/30 bg-yellow-500/10',
    info: 'border-blue-500/30 bg-blue-500/10',
  };

  return (
    <div
      className={cn(
        'flex items-start gap-3 px-4 py-3 rounded-xl border backdrop-blur-xl shadow-lg',
        'animate-slide-up min-w-[300px] max-w-[400px]',
        styles[toast.type],
      )}
      role="alert"
      aria-live="assertive"
      aria-atomic="true"
    >
      <div className="flex-shrink-0 pt-0.5">{icons[toast.type]}</div>
      <div className="flex-1">
        <p className="text-sm font-medium text-white/90">{toast.message}</p>
        {toast.details && <p className="mt-1 text-xs text-white/60">{toast.details}</p>}
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
  );
}
