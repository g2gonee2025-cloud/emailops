/* eslint-disable react-refresh/only-export-components */
import { render, type RenderOptions } from '@testing-library/react';
import { type ReactElement, type ReactNode } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthContext, type AuthContextType } from '../contexts/AuthContext';
import { ToastProvider } from '../components/ui/Toast';

// Create a new QueryClient for each test run
const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: {
      retry: false, // Disable retries for tests
    },
  },
});

// Mock AuthProvider to expose a value setter
const createMockAuthProvider = (authValue: AuthContextType) => {
  return ({ children }: { children: ReactNode }) => (
    <AuthContext.Provider value={authValue}>
      {children}
    </AuthContext.Provider>
  );
};

// Custom render function with all providers
const renderWithProviders = (
  ui: ReactElement,
  {
    authValue = {
      isAuthenticated: false,
      token: null,
      login: vi.fn(),
      logout: vi.fn(),
    },
    ...renderOptions
  }: RenderOptions & { authValue?: AuthContextType } = {},
) => {
  const queryClient = createTestQueryClient();
  const MockAuthProvider = createMockAuthProvider(authValue);

  const Wrapper = ({ children }: { children: ReactNode }) => (
    <MemoryRouter>
      <QueryClientProvider client={queryClient}>
        <ToastProvider>
          <MockAuthProvider>
            {children}
          </MockAuthProvider>
        </ToastProvider>
      </QueryClientProvider>
    </MemoryRouter>
  );

  return render(ui, { wrapper: Wrapper, ...renderOptions });
};

// Re-export everything from testing-library
export * from '@testing-library/react';
// Override render method with our custom one
export { renderWithProviders as render, AuthContext };
