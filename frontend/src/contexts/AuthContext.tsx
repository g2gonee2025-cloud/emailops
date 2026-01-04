/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useState, useEffect, type ReactNode, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '@/lib/api';
import {
  tokenStore,
  setupTokenRefreshTimer,
  TOKEN_EVENTS,
  type TokenSet,
  type TokenResponse,
} from '@/lib/oidc';
import { logger } from '@/lib/logger';

interface AuthContextType {
  isAuthenticated: boolean;
  token: string | null;
  setToken: (token: string | null) => void;
  setTokensFromResponse: (response: TokenResponse) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setTokenState] = useState<string | null>(() => {
    return tokenStore.getAccessToken();
  });
  const navigate = useNavigate();
  const cleanupRefreshTimerRef = useRef<(() => void) | null>(null);

  const isAuthenticated = !!token;

  const logout = useCallback(() => {
    logger.info('Logging out user');
    tokenStore.clearTokens();
    setTokenState(null);
    navigate('/login');
  }, [navigate]);

  const setToken = useCallback((newToken: string | null) => {
    if (newToken) {
      localStorage.setItem('auth_token', newToken);
    } else {
      tokenStore.clearTokens();
    }
    setTokenState(newToken);
  }, []);

  const setTokensFromResponse = useCallback((response: TokenResponse) => {
    const tokens: TokenSet = tokenStore.setTokensFromResponse(response);
    setTokenState(tokens.accessToken);
    api.setAuthToken(tokens.accessToken);
    logger.info('Tokens stored from response', { expiresAt: tokens.expiresAt });
  }, []);

  useEffect(() => {
    if (token) {
      api.setAuthToken(token);

      if (cleanupRefreshTimerRef.current) {
        cleanupRefreshTimerRef.current();
      }
      cleanupRefreshTimerRef.current = setupTokenRefreshTimer(
        '/api/v1/auth/refresh',
        logout,
      );
    } else {
      api.setAuthToken(null);
      if (cleanupRefreshTimerRef.current) {
        cleanupRefreshTimerRef.current();
        cleanupRefreshTimerRef.current = null;
      }
    }

    const handleUnauthorized = () => {
      logout();
    };

    const handleRefreshSuccess = (event: Event) => {
      const customEvent = event as CustomEvent<{ expiresAt?: number }>;
      const newToken = tokenStore.getAccessToken();
      if (newToken) {
        setTokenState(newToken);
        api.setAuthToken(newToken);
        logger.info('Token refreshed successfully', { expiresAt: customEvent.detail?.expiresAt });
      }
    };

    const handleRefreshFailed = () => {
      logger.warn('Token refresh failed, logging out');
      logout();
    };

    window.addEventListener('cortex-unauthorized', handleUnauthorized);
    window.addEventListener(TOKEN_EVENTS.REFRESH_SUCCESS, handleRefreshSuccess);
    window.addEventListener(TOKEN_EVENTS.REFRESH_FAILED, handleRefreshFailed);

    return () => {
      window.removeEventListener('cortex-unauthorized', handleUnauthorized);
      window.removeEventListener(TOKEN_EVENTS.REFRESH_SUCCESS, handleRefreshSuccess);
      window.removeEventListener(TOKEN_EVENTS.REFRESH_FAILED, handleRefreshFailed);
      if (cleanupRefreshTimerRef.current) {
        cleanupRefreshTimerRef.current();
      }
    };
  }, [token, logout]);

  return (
    <AuthContext.Provider value={{ isAuthenticated, token, setToken, setTokensFromResponse, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
