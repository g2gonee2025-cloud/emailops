import { describe, it, expect } from 'vitest';
import {
  ERROR_MAPPING,
  getErrorConfig,
  parseApiErrorResponse,
  shouldRedirectToLogin,
  isInlineError,
  isModalError,
  isToastError,
} from './errors';

describe('Error Mapping', () => {
  describe('ERROR_MAPPING', () => {
    it('should have VALIDATION_ERROR configured as inline pattern', () => {
      expect(ERROR_MAPPING.VALIDATION_ERROR).toEqual({
        pattern: 'inline',
        severity: 'warning',
        retryable: false,
        autoHide: 8000,
      });
    });

    it('should have AUTH_REQUIRED configured as redirect pattern', () => {
      expect(ERROR_MAPPING.AUTH_REQUIRED).toEqual({
        pattern: 'redirect',
        severity: 'error',
        retryable: false,
      });
    });

    it('should have RATE_LIMITED configured as retryable toast', () => {
      expect(ERROR_MAPPING.RATE_LIMITED).toEqual({
        pattern: 'toast',
        severity: 'warning',
        retryable: true,
        actionLabel: 'Retry',
        autoHide: 10000,
      });
    });

    it('should have INTERNAL_ERROR configured as retryable toast', () => {
      expect(ERROR_MAPPING.INTERNAL_ERROR).toEqual({
        pattern: 'toast',
        severity: 'error',
        retryable: true,
        actionLabel: 'Retry',
        autoHide: 10000,
      });
    });

    it('should have CONFIGURATION_ERROR configured as modal', () => {
      expect(ERROR_MAPPING.CONFIGURATION_ERROR).toEqual({
        pattern: 'modal',
        severity: 'critical',
        retryable: false,
      });
    });

    it('should have all expected error codes defined', () => {
      const expectedCodes = [
        'VALIDATION_ERROR',
        'AUTH_REQUIRED',
        'AUTH_SECRET_MISSING',
        'FORBIDDEN',
        'NOT_FOUND',
        'RATE_LIMITED',
        'INTERNAL_ERROR',
        'TRANSACTION_FAILED',
        'RLS_TENANT_REQUIRED',
        'RLS_TENANT_INVALID',
        'RLS_SET_FAILED',
        'DB_URL_MISSING',
        'JSON_SCHEMA_VALIDATION_FAILED',
        'DO_LLM_ENDPOINT_MISSING',
        'INSECURE_JWKS_URL',
        'PROVIDER_ERROR',
        'EMBEDDING_ERROR',
        'RETRIEVAL_ERROR',
        'SECURITY_ERROR',
        'CONFIGURATION_ERROR',
        'FILE_OPERATION_ERROR',
        'CIRCUIT_BREAKER_OPEN',
        'POLICY_VIOLATION',
      ];

      expectedCodes.forEach(code => {
        expect(ERROR_MAPPING).toHaveProperty(code);
      });
    });
  });

  describe('getErrorConfig', () => {
    it('should return config for known error code', () => {
      const config = getErrorConfig('VALIDATION_ERROR');
      expect(config.pattern).toBe('inline');
      expect(config.severity).toBe('warning');
    });

    it('should return default config for unknown error code', () => {
      const config = getErrorConfig('UNKNOWN_ERROR_CODE');
      expect(config.pattern).toBe('toast');
      expect(config.severity).toBe('error');
      expect(config.retryable).toBe(false);
    });

    it('should return default config for null error code', () => {
      const config = getErrorConfig(null);
      expect(config.pattern).toBe('toast');
      expect(config.severity).toBe('error');
    });

    it('should return default config for undefined error code', () => {
      const config = getErrorConfig(undefined);
      expect(config.pattern).toBe('toast');
      expect(config.severity).toBe('error');
    });
  });

  describe('parseApiErrorResponse', () => {
    it('should parse structured error response', () => {
      const details = {
        error: {
          error_code: 'VALIDATION_ERROR',
          message: 'Invalid input',
          correlation_id: 'abc-123',
          retryable: false,
          context: { field: 'email' },
        },
      };

      const parsed = parseApiErrorResponse(details);
      expect(parsed.errorCode).toBe('VALIDATION_ERROR');
      expect(parsed.message).toBe('Invalid input');
      expect(parsed.correlationId).toBe('abc-123');
      expect(parsed.retryable).toBe(false);
      expect(parsed.context).toEqual({ field: 'email' });
    });

    it('should handle legacy detail format', () => {
      const details = {
        detail: 'Something went wrong',
      };

      const parsed = parseApiErrorResponse(details);
      expect(parsed.errorCode).toBeNull();
      expect(parsed.message).toBe('Something went wrong');
      expect(parsed.correlationId).toBeNull();
    });

    it('should handle undefined details', () => {
      const parsed = parseApiErrorResponse(undefined);
      expect(parsed.errorCode).toBeNull();
      expect(parsed.message).toBe('An unexpected error occurred');
      expect(parsed.correlationId).toBeNull();
      expect(parsed.retryable).toBe(false);
    });

    it('should handle empty details object', () => {
      const parsed = parseApiErrorResponse({});
      expect(parsed.errorCode).toBeNull();
      expect(parsed.message).toBe('An unexpected error occurred');
    });
  });

  describe('shouldRedirectToLogin', () => {
    it('should return true for AUTH_REQUIRED', () => {
      expect(shouldRedirectToLogin('AUTH_REQUIRED')).toBe(true);
    });

    it('should return true for RLS_TENANT_REQUIRED', () => {
      expect(shouldRedirectToLogin('RLS_TENANT_REQUIRED')).toBe(true);
    });

    it('should return true for RLS_TENANT_INVALID', () => {
      expect(shouldRedirectToLogin('RLS_TENANT_INVALID')).toBe(true);
    });

    it('should return false for VALIDATION_ERROR', () => {
      expect(shouldRedirectToLogin('VALIDATION_ERROR')).toBe(false);
    });

    it('should return false for null', () => {
      expect(shouldRedirectToLogin(null)).toBe(false);
    });

    it('should return false for undefined', () => {
      expect(shouldRedirectToLogin(undefined)).toBe(false);
    });
  });

  describe('isInlineError', () => {
    it('should return true for VALIDATION_ERROR', () => {
      expect(isInlineError('VALIDATION_ERROR')).toBe(true);
    });

    it('should return false for INTERNAL_ERROR', () => {
      expect(isInlineError('INTERNAL_ERROR')).toBe(false);
    });

    it('should return false for null', () => {
      expect(isInlineError(null)).toBe(false);
    });
  });

  describe('isModalError', () => {
    it('should return true for CONFIGURATION_ERROR', () => {
      expect(isModalError('CONFIGURATION_ERROR')).toBe(true);
    });

    it('should return true for POLICY_VIOLATION', () => {
      expect(isModalError('POLICY_VIOLATION')).toBe(true);
    });

    it('should return false for VALIDATION_ERROR', () => {
      expect(isModalError('VALIDATION_ERROR')).toBe(false);
    });

    it('should return false for null', () => {
      expect(isModalError(null)).toBe(false);
    });
  });

  describe('isToastError', () => {
    it('should return true for INTERNAL_ERROR', () => {
      expect(isToastError('INTERNAL_ERROR')).toBe(true);
    });

    it('should return true for RATE_LIMITED', () => {
      expect(isToastError('RATE_LIMITED')).toBe(true);
    });

    it('should return false for VALIDATION_ERROR (inline)', () => {
      expect(isToastError('VALIDATION_ERROR')).toBe(false);
    });

    it('should return false for AUTH_REQUIRED (redirect)', () => {
      expect(isToastError('AUTH_REQUIRED')).toBe(false);
    });

    it('should return true for null (default is toast)', () => {
      expect(isToastError(null)).toBe(true);
    });

    it('should return true for unknown error code (default is toast)', () => {
      expect(isToastError('UNKNOWN_CODE')).toBe(true);
    });
  });
});
