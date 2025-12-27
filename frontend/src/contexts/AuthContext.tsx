/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useState, useEffect, type ReactNode } from 'react';
import { api } from '../lib/api';

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

  const isAuthenticated = !!token;

  useEffect(() => {
    if (token) {
      localStorage.setItem('auth_token', token);
      api.setAuthToken(token);
    } else {
      localStorage.removeItem('auth_token');
      api.setAuthToken(null);
    }
  }, [token]);

  const login = async (username: string, password: string) => {
    const response = await api.login(username, password);
    setToken(response.access_token);
  };

  const logout = () => {
    setToken(null);
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
