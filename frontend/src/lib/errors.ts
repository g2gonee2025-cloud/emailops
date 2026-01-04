/**
 * Error Mapping and UX Configuration
 *
 * Maps backend error_code values to frontend UX patterns for consistent,
 * actionable error handling across the application.
 */

export type ErrorUXPattern = 'toast' | 'inline' | 'modal' | 'redirect';
export type ErrorSeverity = 'info' | 'warning' | 'error' | 'critical';

export interface ErrorUXConfig {
  pattern: ErrorUXPattern;
  severity: ErrorSeverity;
  retryable: boolean;
  actionLabel?: string;
  autoHide?: number;
}

export interface ParsedApiError {
  errorCode: string | null;
  message: string;
  correlationId: string | null;
  retryable: boolean;
  context?: Record<string, unknown>;
}

export const ERROR_MAPPING: Record<string, ErrorUXConfig> = {
  VALIDATION_ERROR: {
    pattern: 'inline',
    severity: 'warning',
    retryable: false,
    autoHide: 8000,
  },
  AUTH_REQUIRED: {
    pattern: 'redirect',
    severity: 'error',
    retryable: false,
  },
  AUTH_SECRET_MISSING: {
    pattern: 'modal',
    severity: 'critical',
    retryable: false,
  },
  FORBIDDEN: {
    pattern: 'toast',
    severity: 'error',
    retryable: false,
    autoHide: 8000,
  },
  NOT_FOUND: {
    pattern: 'toast',
    severity: 'warning',
    retryable: false,
    autoHide: 6000,
  },
  RATE_LIMITED: {
    pattern: 'toast',
    severity: 'warning',
    retryable: true,
    actionLabel: 'Retry',
    autoHide: 10000,
  },
  INTERNAL_ERROR: {
    pattern: 'toast',
    severity: 'error',
    retryable: true,
    actionLabel: 'Retry',
    autoHide: 10000,
  },
  TRANSACTION_FAILED: {
    pattern: 'toast',
    severity: 'error',
    retryable: true,
    actionLabel: 'Retry',
    autoHide: 10000,
  },
  RLS_TENANT_REQUIRED: {
    pattern: 'redirect',
    severity: 'error',
    retryable: false,
  },
  RLS_TENANT_INVALID: {
    pattern: 'redirect',
    severity: 'error',
    retryable: false,
  },
  RLS_SET_FAILED: {
    pattern: 'toast',
    severity: 'error',
    retryable: true,
    actionLabel: 'Retry',
    autoHide: 10000,
  },
  DB_URL_MISSING: {
    pattern: 'modal',
    severity: 'critical',
    retryable: false,
  },
  JSON_SCHEMA_VALIDATION_FAILED: {
    pattern: 'toast',
    severity: 'error',
    retryable: true,
    actionLabel: 'Retry',
    autoHide: 10000,
  },
  DO_LLM_ENDPOINT_MISSING: {
    pattern: 'modal',
    severity: 'critical',
    retryable: false,
  },
  INSECURE_JWKS_URL: {
    pattern: 'modal',
    severity: 'critical',
    retryable: false,
  },
  PROVIDER_ERROR: {
    pattern: 'toast',
    severity: 'error',
    retryable: true,
    actionLabel: 'Retry',
    autoHide: 10000,
  },
  EMBEDDING_ERROR: {
    pattern: 'toast',
    severity: 'error',
    retryable: true,
    actionLabel: 'Retry',
    autoHide: 10000,
  },
  RETRIEVAL_ERROR: {
    pattern: 'toast',
    severity: 'error',
    retryable: true,
    actionLabel: 'Retry',
    autoHide: 10000,
  },
  SECURITY_ERROR: {
    pattern: 'toast',
    severity: 'error',
    retryable: false,
    autoHide: 10000,
  },
  CONFIGURATION_ERROR: {
    pattern: 'modal',
    severity: 'critical',
    retryable: false,
  },
  FILE_OPERATION_ERROR: {
    pattern: 'toast',
    severity: 'error',
    retryable: true,
    actionLabel: 'Retry',
    autoHide: 8000,
  },
  CIRCUIT_BREAKER_OPEN: {
    pattern: 'toast',
    severity: 'warning',
    retryable: false,
    autoHide: 10000,
  },
  POLICY_VIOLATION: {
    pattern: 'modal',
    severity: 'error',
    retryable: false,
  },
};

const DEFAULT_ERROR_CONFIG: ErrorUXConfig = {
  pattern: 'toast',
  severity: 'error',
  retryable: false,
  autoHide: 8000,
};

export function getErrorConfig(errorCode: string | null | undefined): ErrorUXConfig {
  if (!errorCode) {
    return DEFAULT_ERROR_CONFIG;
  }
  return ERROR_MAPPING[errorCode] ?? DEFAULT_ERROR_CONFIG;
}

export function parseApiErrorResponse(details: Record<string, unknown> | undefined): ParsedApiError {
  if (!details) {
    return {
      errorCode: null,
      message: 'An unexpected error occurred',
      correlationId: null,
      retryable: false,
    };
  }

  const error = details.error as Record<string, unknown> | undefined;

  if (error) {
    return {
      errorCode: (error.error_code as string) ?? null,
      message: (error.message as string) ?? 'An unexpected error occurred',
      correlationId: (error.correlation_id as string) ?? null,
      retryable: (error.retryable as boolean) ?? false,
      context: error.context as Record<string, unknown> | undefined,
    };
  }

  return {
    errorCode: null,
    message: (details.detail as string) ?? 'An unexpected error occurred',
    correlationId: null,
    retryable: false,
  };
}

export function shouldRedirectToLogin(errorCode: string | null | undefined): boolean {
  if (!errorCode) return false;
  const config = getErrorConfig(errorCode);
  return config.pattern === 'redirect';
}

export function isInlineError(errorCode: string | null | undefined): boolean {
  if (!errorCode) return false;
  const config = getErrorConfig(errorCode);
  return config.pattern === 'inline';
}

export function isModalError(errorCode: string | null | undefined): boolean {
  if (!errorCode) return false;
  const config = getErrorConfig(errorCode);
  return config.pattern === 'modal';
}

export function isToastError(errorCode: string | null | undefined): boolean {
  const config = getErrorConfig(errorCode);
  return config.pattern === 'toast';
}
