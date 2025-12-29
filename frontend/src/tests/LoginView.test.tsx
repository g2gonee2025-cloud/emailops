import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { MemoryRouter, useNavigate } from 'react-router-dom';
import { LoginView } from '../components/LoginView';
import { useAuth } from '../contexts/AuthContext';
import { useToast } from '../components/ui/toastContext';

// Mock the hooks
vi.mock('../contexts/AuthContext');
vi.mock('../components/ui/toastContext');
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: vi.fn(),
  };
});

const mockLogin = vi.fn();
const mockAddToast = vi.fn();
const mockNavigate = vi.fn();

describe('LoginView', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (useAuth as vi.Mock).mockReturnValue({ login: mockLogin });
    (useToast as vi.Mock).mockReturnValue({ addToast: mockAddToast });
    (useNavigate as vi.Mock).mockReturnValue(mockNavigate);
  });

  it('navigates to /dashboard on successful login', async () => {
    mockLogin.mockResolvedValue(undefined);

    render(
      <MemoryRouter>
        <LoginView />
      </MemoryRouter>
    );

    fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
    fireEvent.change(screen.getByLabelText(/password/i), { target: { value: 'password' } });
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalledWith('testuser', 'password');
      expect(mockNavigate).toHaveBeenCalledWith('/dashboard');
    });
  });

  it('shows an error toast on failed login', async () => {
    const errorMessage = 'Invalid credentials';
    mockLogin.mockRejectedValue(new Error(errorMessage));

    render(
      <MemoryRouter>
        <LoginView />
      </MemoryRouter>
    );

    fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
    fireEvent.change(screen.getByLabelText(/password/i), { target: { value: 'wrongpassword' } });
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalledWith('testuser', 'wrongpassword');
      expect(mockAddToast).toHaveBeenCalledWith(errorMessage, 'error');
      expect(mockNavigate).not.toHaveBeenCalled();
    });
  });
});
