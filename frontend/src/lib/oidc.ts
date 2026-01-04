/**
 * OIDC Token Management for EmailOps Frontend
 *
 * Handles token storage, expiry detection, and automatic refresh with:
 * - Refresh token storage alongside access token
 * - Token expiry detection with 60s clock skew buffer
 * - Exponential backoff for refresh attempts (max 3 retries)
 * - Guard against concurrent refresh attempts
 * - Event emission on successful token refresh
 */

import { logger } from './logger';

// =============================================================================
// Constants
// =============================================================================

const STORAGE_KEYS = {
  ACCESS_TOKEN: 'auth_token',
  REFRESH_TOKEN: 'refresh_token',
  EXPIRES_AT: 'token_expires_at',
} as const;

const CLOCK_SKEW_BUFFER_SECONDS = 60;
const MAX_REFRESH_RETRIES = 3;
const BASE_RETRY_DELAY_MS = 1000;

// =============================================================================
// Types
// =============================================================================

export interface TokenSet {
  accessToken: string;
  refreshToken: string;
  expiresAt: number;
}

export interface TokenResponse {
  access_token: string;
  refresh_token?: string;
  expires_in: number;
  token_type: string;
}

export interface RefreshResult {
  success: boolean;
  tokens?: TokenSet;
  error?: string;
}

// =============================================================================
// Events
// =============================================================================

export const TOKEN_EVENTS = {
  REFRESH_SUCCESS: 'cortex-token-refresh-success',
  REFRESH_FAILED: 'cortex-token-refresh-failed',
  TOKEN_EXPIRED: 'cortex-token-expired',
} as const;

export function emitTokenEvent(
  eventName: string,
  detail?: Record<string, unknown>,
): void {
  globalThis.dispatchEvent(new CustomEvent(eventName, { detail }));
}

// =============================================================================
// Token Store
// =============================================================================

class TokenStore {
  private refreshPromise: Promise<RefreshResult> | null = null;

  getAccessToken(): string | null {
    return localStorage.getItem(STORAGE_KEYS.ACCESS_TOKEN);
  }

  getRefreshToken(): string | null {
    return localStorage.getItem(STORAGE_KEYS.REFRESH_TOKEN);
  }

  getExpiresAt(): number | null {
    const expiresAt = localStorage.getItem(STORAGE_KEYS.EXPIRES_AT);
    return expiresAt ? parseInt(expiresAt, 10) : null;
  }

  getTokenSet(): TokenSet | null {
    const accessToken = this.getAccessToken();
    const refreshToken = this.getRefreshToken();
    const expiresAt = this.getExpiresAt();

    if (!accessToken || !refreshToken || expiresAt === null) {
      return null;
    }

    return { accessToken, refreshToken, expiresAt };
  }

  setTokens(tokens: TokenSet): void {
    localStorage.setItem(STORAGE_KEYS.ACCESS_TOKEN, tokens.accessToken);
    localStorage.setItem(STORAGE_KEYS.REFRESH_TOKEN, tokens.refreshToken);
    localStorage.setItem(STORAGE_KEYS.EXPIRES_AT, tokens.expiresAt.toString());
  }

  setTokensFromResponse(response: TokenResponse): TokenSet {
    const expiresAt = Date.now() + response.expires_in * 1000;
    const tokens: TokenSet = {
      accessToken: response.access_token,
      refreshToken: response.refresh_token || this.getRefreshToken() || '',
      expiresAt,
    };
    this.setTokens(tokens);
    return tokens;
  }

  clearTokens(): void {
    localStorage.removeItem(STORAGE_KEYS.ACCESS_TOKEN);
    localStorage.removeItem(STORAGE_KEYS.REFRESH_TOKEN);
    localStorage.removeItem(STORAGE_KEYS.EXPIRES_AT);
  }

  isTokenExpired(): boolean {
    const expiresAt = this.getExpiresAt();
    if (expiresAt === null) {
      return true;
    }
    const bufferMs = CLOCK_SKEW_BUFFER_SECONDS * 1000;
    return Date.now() >= expiresAt - bufferMs;
  }

  isTokenExpiringSoon(bufferSeconds: number = CLOCK_SKEW_BUFFER_SECONDS): boolean {
    const expiresAt = this.getExpiresAt();
    if (expiresAt === null) {
      return true;
    }
    const bufferMs = bufferSeconds * 1000;
    return Date.now() >= expiresAt - bufferMs;
  }

  getTimeUntilExpiry(): number | null {
    const expiresAt = this.getExpiresAt();
    if (expiresAt === null) {
      return null;
    }
    return Math.max(0, expiresAt - Date.now());
  }

  hasRefreshInProgress(): boolean {
    return this.refreshPromise !== null;
  }

  setRefreshPromise(promise: Promise<RefreshResult> | null): void {
    this.refreshPromise = promise;
  }

  getRefreshPromise(): Promise<RefreshResult> | null {
    return this.refreshPromise;
  }
}

export const tokenStore = new TokenStore();

// =============================================================================
// Token Refresh Logic
// =============================================================================

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function calculateBackoffDelay(attempt: number): number {
  return BASE_RETRY_DELAY_MS * Math.pow(2, attempt);
}

async function performTokenRefresh(
  refreshToken: string,
  tokenEndpoint: string,
): Promise<TokenResponse> {
  const response = await fetch(tokenEndpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      grant_type: 'refresh_token',
      refresh_token: refreshToken,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.error_description ||
        errorData.detail ||
        `Token refresh failed with status ${response.status}`,
    );
  }

  return response.json();
}

export async function refreshAccessToken(
  tokenEndpoint: string = '/api/v1/auth/refresh',
): Promise<RefreshResult> {
  if (tokenStore.hasRefreshInProgress()) {
    logger.debug('Token refresh already in progress, waiting for existing refresh');
    const existingPromise = tokenStore.getRefreshPromise();
    if (existingPromise) {
      return existingPromise;
    }
  }

  const refreshPromise = executeRefreshWithRetry(tokenEndpoint);
  tokenStore.setRefreshPromise(refreshPromise);

  try {
    const result = await refreshPromise;
    return result;
  } finally {
    tokenStore.setRefreshPromise(null);
  }
}

async function executeRefreshWithRetry(
  tokenEndpoint: string,
): Promise<RefreshResult> {
  const refreshToken = tokenStore.getRefreshToken();

  if (!refreshToken) {
    logger.warn('No refresh token available');
    emitTokenEvent(TOKEN_EVENTS.REFRESH_FAILED, { reason: 'no_refresh_token' });
    return { success: false, error: 'No refresh token available' };
  }

  let lastError: Error | null = null;

  for (let attempt = 0; attempt < MAX_REFRESH_RETRIES; attempt++) {
    try {
      if (attempt > 0) {
        const delay = calculateBackoffDelay(attempt - 1);
        logger.debug(`Retry attempt ${attempt + 1}/${MAX_REFRESH_RETRIES}, waiting ${delay}ms`);
        await sleep(delay);
      }

      const tokenResponse = await performTokenRefresh(refreshToken, tokenEndpoint);
      const tokens = tokenStore.setTokensFromResponse(tokenResponse);

      logger.info('Token refresh successful');
      emitTokenEvent(TOKEN_EVENTS.REFRESH_SUCCESS, {
        expiresAt: tokens.expiresAt,
        attempt: attempt + 1,
      });

      return { success: true, tokens };
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      logger.warn(`Token refresh attempt ${attempt + 1} failed: ${lastError.message}`);

      if (
        lastError.message.includes('invalid_grant') ||
        lastError.message.includes('expired') ||
        lastError.message.includes('401')
      ) {
        logger.error('Refresh token is invalid or expired, stopping retries');
        break;
      }
    }
  }

  const errorMessage = lastError?.message || 'Token refresh failed after all retries';
  logger.error(`Token refresh failed: ${errorMessage}`);
  emitTokenEvent(TOKEN_EVENTS.REFRESH_FAILED, {
    reason: 'refresh_failed',
    error: errorMessage,
  });

  return { success: false, error: errorMessage };
}

// =============================================================================
// Token Validation & Auto-Refresh
// =============================================================================

export async function ensureValidToken(
  tokenEndpoint?: string,
): Promise<string | null> {
  const accessToken = tokenStore.getAccessToken();

  if (!accessToken) {
    return null;
  }

  if (!tokenStore.isTokenExpired()) {
    return accessToken;
  }

  logger.info('Access token expired or expiring soon, attempting refresh');
  emitTokenEvent(TOKEN_EVENTS.TOKEN_EXPIRED);

  const result = await refreshAccessToken(tokenEndpoint);

  if (result.success && result.tokens) {
    return result.tokens.accessToken;
  }

  return null;
}

export function setupTokenRefreshTimer(
  tokenEndpoint?: string,
  onRefreshFailed?: () => void,
): () => void {
  let timeoutId: ReturnType<typeof setTimeout> | null = null;

  const scheduleRefresh = () => {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }

    const timeUntilExpiry = tokenStore.getTimeUntilExpiry();
    if (timeUntilExpiry === null) {
      return;
    }

    const refreshTime = Math.max(
      0,
      timeUntilExpiry - CLOCK_SKEW_BUFFER_SECONDS * 1000 - 5000,
    );

    logger.debug(`Scheduling token refresh in ${refreshTime}ms`);

    timeoutId = setTimeout(async () => {
      const result = await refreshAccessToken(tokenEndpoint);
      if (result.success) {
        scheduleRefresh();
      } else if (onRefreshFailed) {
        onRefreshFailed();
      }
    }, refreshTime);
  };

  scheduleRefresh();

  const handleRefreshSuccess = () => {
    scheduleRefresh();
  };

  globalThis.addEventListener(TOKEN_EVENTS.REFRESH_SUCCESS, handleRefreshSuccess);

  return () => {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    globalThis.removeEventListener(TOKEN_EVENTS.REFRESH_SUCCESS, handleRefreshSuccess);
  };
}

// =============================================================================
// Utility Functions
// =============================================================================

export function parseJwtExpiry(token: string): number | null {
  try {
    const parts = token.split('.');
    if (parts.length !== 3) {
      return null;
    }
    const payload = JSON.parse(atob(parts[1]));
    if (typeof payload.exp === 'number') {
      return payload.exp * 1000;
    }
    return null;
  } catch {
    return null;
  }
}

export function initializeTokensFromJwt(
  accessToken: string,
  refreshToken: string,
): TokenSet | null {
  const expiresAt = parseJwtExpiry(accessToken);
  if (expiresAt === null) {
    return null;
  }

  const tokens: TokenSet = {
    accessToken,
    refreshToken,
    expiresAt,
  };

  tokenStore.setTokens(tokens);
  return tokens;
}
