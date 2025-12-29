import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import App from '../App';
import { useAuth } from '../contexts/AuthContext';

// Mock the useAuth hook
vi.mock('../contexts/AuthContext', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../contexts/AuthContext')>();
  return {
    ...actual,
    useAuth: vi.fn(),
  };
});

describe('App Routing', () => {
  test('redirects to /login from /dashboard when unauthenticated', async () => {
    // Arrange: Mock useAuth to return unauthenticated state
    (useAuth as vi.Mock).mockReturnValue({ isAuthenticated: false });

    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <App />
      </MemoryRouter>
    );

    // Assert: Wait for the redirect and check for an element on the login page
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /EmailOps Cortex/i })).toBeInTheDocument();
    });
  });

  test('renders protected layout for /thread/:id when authenticated', () => {
    // Arrange: Mock useAuth to return authenticated state
    (useAuth as vi.Mock).mockReturnValue({ isAuthenticated: true });

    render(
      <MemoryRouter initialEntries={['/thread/some-cool-thread-id']}>
        <App />
      </MemoryRouter>
    );

    // Assert: The main layout should be rendered.
    expect(screen.getByRole('navigation', { name: /main navigation/i })).toBeInTheDocument();
  });
});
