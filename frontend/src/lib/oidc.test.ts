import { describe, it, expect, beforeEach, afterEach, vi, type Mock } from 'vitest';
import {
  tokenStore,
  refreshAccessToken,
  ensureValidToken,
  parseJwtExpiry,
  initializeTokensFromJwt,
  TOKEN_EVENTS,
  emitTokenEvent,
  type TokenSet,
  type TokenResponse,
} from './oidc';

vi.mock('./logger', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

describe('OIDC Token Management', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('TokenStore', () => {
    describe('getAccessToken', () => {
      it('should return null when no token is stored', () => {
        expect(tokenStore.getAccessToken()).toBeNull();
      });

      it('should return the stored access token', () => {
        localStorage.setItem('auth_token', 'test-access-token');
        expect(tokenStore.getAccessToken()).toBe('test-access-token');
      });
    });

    describe('getRefreshToken', () => {
      it('should return null when no refresh token is stored', () => {
        expect(tokenStore.getRefreshToken()).toBeNull();
      });

      it('should return the stored refresh token', () => {
        localStorage.setItem('refresh_token', 'test-refresh-token');
        expect(tokenStore.getRefreshToken()).toBe('test-refresh-token');
      });
    });

    describe('getExpiresAt', () => {
      it('should return null when no expiry is stored', () => {
        expect(tokenStore.getExpiresAt()).toBeNull();
      });

      it('should return the stored expiry timestamp', () => {
        const expiresAt = Date.now() + 3600000;
        localStorage.setItem('token_expires_at', expiresAt.toString());
        expect(tokenStore.getExpiresAt()).toBe(expiresAt);
      });
    });

    describe('getTokenSet', () => {
      it('should return null when tokens are incomplete', () => {
        localStorage.setItem('auth_token', 'test-access-token');
        expect(tokenStore.getTokenSet()).toBeNull();
      });

      it('should return complete token set when all tokens are stored', () => {
        const expiresAt = Date.now() + 3600000;
        localStorage.setItem('auth_token', 'test-access-token');
        localStorage.setItem('refresh_token', 'test-refresh-token');
        localStorage.setItem('token_expires_at', expiresAt.toString());

        const tokenSet = tokenStore.getTokenSet();
        expect(tokenSet).toEqual({
          accessToken: 'test-access-token',
          refreshToken: 'test-refresh-token',
          expiresAt,
        });
      });
    });

    describe('setTokens', () => {
      it('should store all tokens in localStorage', () => {
        const tokens: TokenSet = {
          accessToken: 'new-access-token',
          refreshToken: 'new-refresh-token',
          expiresAt: Date.now() + 3600000,
        };

        tokenStore.setTokens(tokens);

        expect(localStorage.getItem('auth_token')).toBe('new-access-token');
        expect(localStorage.getItem('refresh_token')).toBe('new-refresh-token');
        expect(localStorage.getItem('token_expires_at')).toBe(tokens.expiresAt.toString());
      });
    });

    describe('setTokensFromResponse', () => {
      it('should convert token response to token set and store it', () => {
        const now = Date.now();
        vi.setSystemTime(now);

        const response: TokenResponse = {
          access_token: 'response-access-token',
          refresh_token: 'response-refresh-token',
          expires_in: 3600,
          token_type: 'Bearer',
        };

        const tokens = tokenStore.setTokensFromResponse(response);

        expect(tokens.accessToken).toBe('response-access-token');
        expect(tokens.refreshToken).toBe('response-refresh-token');
        expect(tokens.expiresAt).toBe(now + 3600000);
      });

      it('should preserve existing refresh token if not provided in response', () => {
        localStorage.setItem('refresh_token', 'existing-refresh-token');

        const response: TokenResponse = {
          access_token: 'new-access-token',
          expires_in: 3600,
          token_type: 'Bearer',
        };

        const tokens = tokenStore.setTokensFromResponse(response);
        expect(tokens.refreshToken).toBe('existing-refresh-token');
      });
    });

    describe('clearTokens', () => {
      it('should remove all tokens from localStorage', () => {
        localStorage.setItem('auth_token', 'test-access-token');
        localStorage.setItem('refresh_token', 'test-refresh-token');
        localStorage.setItem('token_expires_at', '123456789');

        tokenStore.clearTokens();

        expect(localStorage.getItem('auth_token')).toBeNull();
        expect(localStorage.getItem('refresh_token')).toBeNull();
        expect(localStorage.getItem('token_expires_at')).toBeNull();
      });
    });

    describe('isTokenExpired', () => {
      it('should return true when no expiry is stored', () => {
        expect(tokenStore.isTokenExpired()).toBe(true);
      });

      it('should return true when token is expired', () => {
        const now = Date.now();
        vi.setSystemTime(now);
        localStorage.setItem('token_expires_at', (now - 1000).toString());
        expect(tokenStore.isTokenExpired()).toBe(true);
      });

      it('should return true when token expires within clock skew buffer (60s)', () => {
        const now = Date.now();
        vi.setSystemTime(now);
        localStorage.setItem('token_expires_at', (now + 30000).toString());
        expect(tokenStore.isTokenExpired()).toBe(true);
      });

      it('should return false when token is valid and not expiring soon', () => {
        const now = Date.now();
        vi.setSystemTime(now);
        localStorage.setItem('token_expires_at', (now + 120000).toString());
        expect(tokenStore.isTokenExpired()).toBe(false);
      });
    });

    describe('isTokenExpiringSoon', () => {
      it('should return true when token expires within custom buffer', () => {
        const now = Date.now();
        vi.setSystemTime(now);
        localStorage.setItem('token_expires_at', (now + 90000).toString());
        expect(tokenStore.isTokenExpiringSoon(120)).toBe(true);
      });

      it('should return false when token is valid beyond custom buffer', () => {
        const now = Date.now();
        vi.setSystemTime(now);
        localStorage.setItem('token_expires_at', (now + 180000).toString());
        expect(tokenStore.isTokenExpiringSoon(120)).toBe(false);
      });
    });

    describe('getTimeUntilExpiry', () => {
      it('should return null when no expiry is stored', () => {
        expect(tokenStore.getTimeUntilExpiry()).toBeNull();
      });

      it('should return time until expiry in milliseconds', () => {
        const now = Date.now();
        vi.setSystemTime(now);
        localStorage.setItem('token_expires_at', (now + 60000).toString());
        expect(tokenStore.getTimeUntilExpiry()).toBe(60000);
      });

      it('should return 0 when token is already expired', () => {
        const now = Date.now();
        vi.setSystemTime(now);
        localStorage.setItem('token_expires_at', (now - 1000).toString());
        expect(tokenStore.getTimeUntilExpiry()).toBe(0);
      });
    });
  });

  describe('Token Refresh', () => {
    beforeEach(() => {
      global.fetch = vi.fn();
    });

    describe('refreshAccessToken', () => {
      it('should return error when no refresh token is available', async () => {
        const result = await refreshAccessToken();
        expect(result.success).toBe(false);
        expect(result.error).toBe('No refresh token available');
      });

      it('should successfully refresh tokens', async () => {
        const now = Date.now();
        vi.setSystemTime(now);

        localStorage.setItem('refresh_token', 'valid-refresh-token');

        (fetch as Mock).mockResolvedValue({
          ok: true,
          json: async () => ({
            access_token: 'new-access-token',
            refresh_token: 'new-refresh-token',
            expires_in: 3600,
            token_type: 'Bearer',
          }),
        });

        const eventSpy = vi.fn();
        globalThis.addEventListener(TOKEN_EVENTS.REFRESH_SUCCESS, eventSpy);

        const result = await refreshAccessToken('/api/v1/auth/refresh');

        expect(result.success).toBe(true);
        expect(result.tokens?.accessToken).toBe('new-access-token');
        expect(result.tokens?.refreshToken).toBe('new-refresh-token');
        expect(eventSpy).toHaveBeenCalled();

        globalThis.removeEventListener(TOKEN_EVENTS.REFRESH_SUCCESS, eventSpy);
      });

      it('should retry with exponential backoff on failure', async () => {
        localStorage.setItem('refresh_token', 'valid-refresh-token');

        (fetch as Mock)
          .mockRejectedValueOnce(new Error('Network error'))
          .mockRejectedValueOnce(new Error('Network error'))
          .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
              access_token: 'new-access-token',
              refresh_token: 'new-refresh-token',
              expires_in: 3600,
              token_type: 'Bearer',
            }),
          });

        const resultPromise = refreshAccessToken('/api/v1/auth/refresh');

        await vi.advanceTimersByTimeAsync(1000);
        await vi.advanceTimersByTimeAsync(2000);

        const result = await resultPromise;

        expect(result.success).toBe(true);
        expect(fetch).toHaveBeenCalledTimes(3);
      });

      it('should stop retrying on invalid_grant error', async () => {
        localStorage.setItem('refresh_token', 'invalid-refresh-token');

        (fetch as Mock).mockResolvedValue({
          ok: false,
          status: 400,
          json: async () => ({
            error: 'invalid_grant',
            error_description: 'Refresh token is invalid or expired',
          }),
        });

        const eventSpy = vi.fn();
        globalThis.addEventListener(TOKEN_EVENTS.REFRESH_FAILED, eventSpy);

        const result = await refreshAccessToken('/api/v1/auth/refresh');

        expect(result.success).toBe(false);
        expect(fetch).toHaveBeenCalledTimes(1);
        expect(eventSpy).toHaveBeenCalled();

        globalThis.removeEventListener(TOKEN_EVENTS.REFRESH_FAILED, eventSpy);
      });

      it('should prevent concurrent refresh attempts', async () => {
        localStorage.setItem('refresh_token', 'valid-refresh-token');

        let resolveFirst: (value: Response) => void;
        const firstPromise = new Promise<Response>((resolve) => {
          resolveFirst = resolve;
        });

        (fetch as Mock).mockReturnValueOnce(firstPromise);

        const refresh1 = refreshAccessToken('/api/v1/auth/refresh');
        const refresh2 = refreshAccessToken('/api/v1/auth/refresh');

        resolveFirst!({
          ok: true,
          json: async () => ({
            access_token: 'new-access-token',
            refresh_token: 'new-refresh-token',
            expires_in: 3600,
            token_type: 'Bearer',
          }),
        } as Response);

        const [result1, result2] = await Promise.all([refresh1, refresh2]);

        expect(result1).toEqual(result2);
        expect(fetch).toHaveBeenCalledTimes(1);
      });
    });

    describe('ensureValidToken', () => {
      it('should return null when no access token exists', async () => {
        const result = await ensureValidToken();
        expect(result).toBeNull();
      });

      it('should return existing token when not expired', async () => {
        const now = Date.now();
        vi.setSystemTime(now);

        localStorage.setItem('auth_token', 'valid-access-token');
        localStorage.setItem('token_expires_at', (now + 120000).toString());

        const result = await ensureValidToken();
        expect(result).toBe('valid-access-token');
      });

      it('should refresh token when expired', async () => {
        const now = Date.now();
        vi.setSystemTime(now);

        localStorage.setItem('auth_token', 'expired-access-token');
        localStorage.setItem('refresh_token', 'valid-refresh-token');
        localStorage.setItem('token_expires_at', (now - 1000).toString());

        (fetch as Mock).mockResolvedValue({
          ok: true,
          json: async () => ({
            access_token: 'new-access-token',
            refresh_token: 'new-refresh-token',
            expires_in: 3600,
            token_type: 'Bearer',
          }),
        });

        const result = await ensureValidToken('/api/v1/auth/refresh');
        expect(result).toBe('new-access-token');
      });

      it('should return null when refresh fails', async () => {
        const now = Date.now();
        vi.setSystemTime(now);

        localStorage.setItem('auth_token', 'expired-access-token');
        localStorage.setItem('refresh_token', 'invalid-refresh-token');
        localStorage.setItem('token_expires_at', (now - 1000).toString());

        (fetch as Mock).mockResolvedValue({
          ok: false,
          status: 400,
          json: async () => ({ error: 'invalid_grant', error_description: 'Refresh token is invalid_grant or expired' }),
        });

        const result = await ensureValidToken('/api/v1/auth/refresh');
        expect(result).toBeNull();
      });
    });
  });

  describe('Utility Functions', () => {
    describe('parseJwtExpiry', () => {
      it('should return null for invalid JWT format', () => {
        expect(parseJwtExpiry('invalid')).toBeNull();
        expect(parseJwtExpiry('a.b')).toBeNull();
        expect(parseJwtExpiry('')).toBeNull();
      });

      it('should return null for JWT without exp claim', () => {
        const payload = btoa(JSON.stringify({ sub: 'user123' }));
        const token = `header.${payload}.signature`;
        expect(parseJwtExpiry(token)).toBeNull();
      });

      it('should return expiry timestamp from valid JWT', () => {
        const exp = Math.floor(Date.now() / 1000) + 3600;
        const payload = btoa(JSON.stringify({ sub: 'user123', exp }));
        const token = `header.${payload}.signature`;
        expect(parseJwtExpiry(token)).toBe(exp * 1000);
      });
    });

    describe('initializeTokensFromJwt', () => {
      it('should return null for invalid JWT', () => {
        const result = initializeTokensFromJwt('invalid', 'refresh-token');
        expect(result).toBeNull();
      });

      it('should initialize tokens from valid JWT', () => {
        const exp = Math.floor(Date.now() / 1000) + 3600;
        const payload = btoa(JSON.stringify({ sub: 'user123', exp }));
        const accessToken = `header.${payload}.signature`;
        const refreshToken = 'test-refresh-token';

        const result = initializeTokensFromJwt(accessToken, refreshToken);

        expect(result).not.toBeNull();
        expect(result?.accessToken).toBe(accessToken);
        expect(result?.refreshToken).toBe(refreshToken);
        expect(result?.expiresAt).toBe(exp * 1000);
        expect(localStorage.getItem('auth_token')).toBe(accessToken);
        expect(localStorage.getItem('refresh_token')).toBe(refreshToken);
      });
    });

    describe('emitTokenEvent', () => {
      it('should dispatch custom event with detail', () => {
        const eventSpy = vi.fn();
        globalThis.addEventListener('test-event', eventSpy);

        emitTokenEvent('test-event', { foo: 'bar' });

        expect(eventSpy).toHaveBeenCalled();
        const event = eventSpy.mock.calls[0][0] as CustomEvent;
        expect(event.detail).toEqual({ foo: 'bar' });

        globalThis.removeEventListener('test-event', eventSpy);
      });
    });
  });
});
