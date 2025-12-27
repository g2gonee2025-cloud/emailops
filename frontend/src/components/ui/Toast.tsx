import { useState, useEffect, useCallback } from 'react';
import { cn } from '../../lib/utils';
import { CheckCircle, AlertCircle, Info, X, AlertTriangle } from 'lucide-react';
import { ToastContext, type ToastContextType, type Toast } from './toastContext';

// Container (renders all toasts)
function ToastContainer({ toasts, removeToast }: { toasts: Toast[]; removeToast: (id: string) => void }) {
  return (
    <div
      className="fixed bottom-6 right-6 z-50 flex flex-col gap-3"
      role="region"
      aria-label="Notifications"
    >
      {toasts.map(toast => (
        <ToastItem key={toast.id} toast={toast} onRemove={removeToast} />
      ))}
    </div>
  );
}

// Individual toast item
function ToastItem({ toast, onRemove }: { toast: Toast; onRemove: (id: string) => void }) {
  useEffect(() => {
    if (toast.duration && toast.duration > 0) {
      const timer = setTimeout(() => onRemove(toast.id), toast.duration);
      return () => clearTimeout(timer);
    }
  }, [toast.id, toast.duration, onRemove]);

  const icons = {
    success: <CheckCircle className="w-5 h-5 text-green-400" />,
    error: <AlertCircle className="w-5 h-5 text-red-400" />,
    warning: <AlertTriangle className="w-5 h-5 text-yellow-400" />,
    info: <Info className="w-5 h-5 text-blue-400" />,
  };

  const styles = {
    success: 'border-green-500/30 bg-green-500/10',
    error: 'border-red-500/30 bg-red-500/10',
    warning: 'border-yellow-500/30 bg-yellow-500/10',
    info: 'border-blue-500/30 bg-blue-500/10',
  };

  return (
    <div
      className={cn(
        "flex items-center gap-3 px-4 py-3 rounded-xl border backdrop-blur-xl shadow-lg",
        "animate-slide-up min-w-[300px] max-w-[400px]",
        styles[toast.type]
      )}
      role="alert"
      aria-live="polite"
    >
      {icons[toast.type]}
      <p className="flex-1 text-sm text-white/90">{toast.message}</p>
      <button
        onClick={() => onRemove(toast.id)}
        className="p-1 rounded hover:bg-white/10 transition-colors"
        aria-label="Dismiss notification"
      >
        <X className="w-4 h-4 text-white/50" />
      </button>
    </div>
  );
}

// Provider - only component export from this file
export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const addToast = useCallback((message: string, type: Toast['type'] = 'info', duration = 5000) => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    setToasts(prev => [...prev, { id, message, type, duration }]);
  }, []);

  const removeToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  const value: ToastContextType = { toasts, addToast, removeToast };

  return (
    <ToastContext.Provider value={value}>
      {children}
      <ToastContainer toasts={toasts} removeToast={removeToast} />
    </ToastContext.Provider>
  );
}
