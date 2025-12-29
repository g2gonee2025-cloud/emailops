/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useState, useEffect, type ReactNode, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '@/lib/api';

interface AuthContextType {
  isAuthenticated: boolean;
  token: string | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(() => {
    return localStorage.getItem('auth_token');
  });
  const navigate = useNavigate();

  const isAuthenticated = !!token;

  const logout = useCallback(() => {
    setToken(null);
    navigate('/login');
  }, [navigate]);

  useEffect(() => {
    if (token) {
      localStorage.setItem('auth_token', token);
      api.setAuthToken(token);
    } else {
      localStorage.removeItem('auth_token');
      api.setAuthToken(null);
    }

    const handleUnauthorized = () => {
      logout();
    };

    window.addEventListener('cortex-unauthorized', handleUnauthorized);

    return () => {
      window.removeEventListener('cortex-unauthorized', handleUnauthorized);
    };
  }, [token, logout]);

  const login = async (username: string, password: string) => {
    try {
      const response = await api.login(username, password);
      setToken(response.access_token);
    } catch (error) {
      // Re-throw other errors to be handled by the UI component
      console.error('Login failed:', error);
      throw error;
    }
  };

  return (
    <AuthContext.Provider value={{ isAuthenticated, token, login, logout }}>
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
