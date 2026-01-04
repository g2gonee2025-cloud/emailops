import { render, screen, fireEvent, waitFor } from '../tests/testUtils';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach, type Mock } from 'vitest';
import LoginView from './LoginView';
import * as useLoginHook from '../hooks/useLogin';
import * as AuthContext from '../contexts/AuthContext';
import * as ReactRouterDom from 'react-router-dom';

// Mocking hooks and router
vi.mock('../hooks/useLogin');
vi.mock('../contexts/AuthContext', async () => {
    const actual = await vi.importActual('../contexts/AuthContext');
    return {
      ...actual as object,
      useAuth: vi.fn(),
    };
  });

  vi.mock('react-router-dom', async () => {
      const actual = await vi.importActual('react-router-dom');
      return {
        ...actual as object,
        useNavigate: vi.fn(),
      };
    });

describe('LoginView', () => {
  const mockUseLogin = useLoginHook.useLogin as Mock;
  const mockUseAuth = AuthContext.useAuth as Mock;
  const mockUseNavigate = ReactRouterDom.useNavigate as Mock;

  let mockLoginAsync: Mock;
  let mockSetToken: Mock;
  let mockNavigate: Mock;

  beforeEach(() => {
    vi.clearAllMocks();
    mockLoginAsync = vi.fn();
    mockSetToken = vi.fn();
    mockNavigate = vi.fn();

    mockUseLogin.mockReturnValue({
      loginAsync: mockLoginAsync,
      error: null,
      isLoading: false,
      isSuccess: false,
      data: null,
    });

    mockUseAuth.mockReturnValue({
      setToken: mockSetToken,
      token: null,
      isAuthenticated: false,
      logout: vi.fn(),
    });

    mockUseNavigate.mockReturnValue(mockNavigate);
  });

  it('renders the login form correctly', () => {
    render(<LoginView />);
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
  });

  it('shows loading state when submitting', () => {
    mockUseLogin.mockReturnValue({
      ...mockUseLogin(),
      isLoading: true,
    });
    render(<LoginView />);
    const button = screen.getByRole('button', { name: /signing in.../i });
    expect(button).toBeInTheDocument();
    expect(button).toBeDisabled();
  });

  it('displays an error message on failed login', async () => {
    const error = new Error('Invalid credentials');
    mockLoginAsync.mockRejectedValue(error);
    const { rerender } = render(<LoginView />);

    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: 'test@example.com' } });
    fireEvent.change(screen.getByLabelText(/password/i), { target: { value: 'password' } });
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() => expect(mockLoginAsync).toHaveBeenCalled());

    mockUseLogin.mockReturnValue({
        ...mockUseLogin(),
        error,
    });
    rerender(<LoginView />);

    await waitFor(() => {
      expect(screen.getByText('Invalid credentials')).toBeInTheDocument();
    });
  });

  it('handles successful login and navigation', async () => {
    const token = { access_token: 'fake-token', refresh_token: 'fake-refresh', expires_in: 3600 };
    mockLoginAsync.mockResolvedValue(token);
    const { rerender } = render(<LoginView />);

    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: 'test@example.com' } });
    fireEvent.change(screen.getByLabelText(/password/i), { target: { value: 'password123' } });
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() => expect(mockLoginAsync).toHaveBeenCalledWith(['test@example.com', 'password123']));

    mockUseLogin.mockReturnValue({
      ...mockUseLogin(),
      isSuccess: true,
      data: token,
    });
    rerender(<LoginView />);

    // LoginView now only navigates on success - token storage is handled by useLogin hook
    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/dashboard');
    });
  });

  it('displays validation error for empty fields', async () => {
    render(<LoginView />);
    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: '' } });
    fireEvent.change(screen.getByLabelText(/password/i), { target: { value: '' } });
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() => {
      expect(screen.getByText('Email is required')).toBeInTheDocument();
      expect(screen.getByText('Password is required')).toBeInTheDocument();
    });
  });

  it('displays validation error for invalid email format', async () => {
    const user = userEvent.setup();
    render(<LoginView />);

    // Clear the input first just in case
    const emailInput = screen.getByLabelText(/email/i);
    await user.clear(emailInput);
    await user.type(emailInput, 'not-an-email');

    await user.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() => {
      expect(screen.getByText('Invalid email address')).toBeInTheDocument();
    });
  });
});
