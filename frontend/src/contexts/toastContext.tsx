/* eslint-disable react-refresh/only-export-components */
import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { type ApiError } from '../lib/api';
import { ToastContainer } from '../components/ui/Toast';

// =============================================================================
// Type Definitions
// =============================================================================

export type ToastType = 'success' | 'error' | 'info' | 'warning';

export interface Toast {
  id: string;
  message: string;
  type: ToastType;
  duration?: number;
  details?: string;
}

export interface AddToastParams {
  message: string;
  type?: ToastType;
  duration?: number;
  details?: string;
}

export interface ToastContextType {
  toasts: Toast[];
  addToast: (params: AddToastParams) => void;
  removeToast: (id: string) => void;
}

// =============================================================================
// Context & Hook
// =============================================================================

export const ToastContext = createContext<ToastContextType | undefined>(undefined);

export function useToast(): ToastContextType {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
}

// =============================================================================
// Provider Component
// =============================================================================

const MAX_TOAST_DURATION = 15000;

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const addToast = useCallback(
    ({ message, type = 'info', duration = 5000, details }: AddToastParams) => {
      const id = `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      const cappedDuration = Math.min(duration, MAX_TOAST_DURATION);
      setToasts(prev => [...prev, { id, message, type, duration: cappedDuration, details }]);
    },
    [],
  );

  const removeToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  useEffect(() => {
    const handleApiError = (event: Event) => {
      const customEvent = event as CustomEvent<ApiError>;
      const error = customEvent.detail;

      console.error('Global API Error:', error);

      const isDev = import.meta.env.DEV;
      const message = (error.details?.detail as string) || (isDev ? error.message : 'An unexpected API error occurred.');
      const details = isDev ? `[${error.status}]` : undefined;

      addToast({ message, details, type: 'error', duration: 10000 });
    };

    window.addEventListener('api:error', handleApiError);
    return () => {
      window.removeEventListener('api:error', handleApiError);
    };
  }, [addToast]);

  const value: ToastContextType = { toasts, addToast, removeToast };

  return (
    <ToastContext.Provider value={value}>
      {children}
      <ToastContainer toasts={toasts} removeToast={removeToast} />
    </ToastContext.Provider>
  );
}
