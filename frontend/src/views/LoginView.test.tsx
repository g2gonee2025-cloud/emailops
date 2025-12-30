import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import LoginView from './LoginView';
import { AuthProvider } from '../contexts/AuthContext';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import * as useLogin from '../hooks/useLogin';

// Mock the useLogin hook
vi.mock('../hooks/useLogin', () => ({
  useLogin: vi.fn(),
}));

const queryClient = new QueryClient();

const renderComponent = () => {
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <AuthProvider>
          <LoginView />
        </AuthProvider>
      </MemoryRouter>
    </QueryClientProvider>
  );
};

describe('LoginView', () => {
  it('renders the login form', () => {
    (useLogin.useLogin as vi.Mock).mockReturnValue({
      loginAsync: vi.fn(),
      error: null,
      isLoading: false,
      isSuccess: false,
      data: null,
    });

    renderComponent();

    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
  });

  it('handles successful login', async () => {
    const loginAsync = vi.fn().mockResolvedValue({ access_token: 'test_token' });
    (useLogin.useLogin as vi.Mock).mockReturnValue({
      loginAsync,
      error: null,
      isLoading: false,
      isSuccess: true,
      data: { access_token: 'test_token' },
    });

    renderComponent();

    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: 'test@example.com' } });
    fireEvent.change(screen.getByLabelText(/password/i), { target: { value: 'password' } });
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() => {
      expect(loginAsync).toHaveBeenCalledWith(['test@example.com', 'password']);
    });
  });

  it('shows loading state when submitting', () => {
    (useLogin.useLogin as vi.Mock).mockReturnValue({
      loginAsync: vi.fn(),
      error: null,
      isLoading: true,
      isSuccess: false,
      data: null,
    });

    renderComponent();

    expect(screen.getByRole('button', { name: /signing in.../i })).toBeDisabled();
  });

  it('shows an error message on failed login', async () => {
    const loginAsync = vi.fn().mockRejectedValue(new Error('Invalid credentials'));
    (useLogin.useLogin as vi.Mock).mockReturnValue({
      loginAsync,
      error: new Error('Invalid credentials'),
      isLoading: false,
      isSuccess: false,
      data: null,
    });

    renderComponent();

    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: 'test@example.com' } });
    fireEvent.change(screen.getByLabelText(/password/i), { target: { value: 'password' } });
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() => {
      expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
    });
  });
});
