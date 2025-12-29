// App.test.tsx
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import App from './App';
import { useAuth } from './contexts/AuthContext';

// Mock the useAuth hook
vi.mock('./contexts/AuthContext', () => ({
  useAuth: vi.fn(),
  AuthProvider: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));


describe('App', () => {
  it('renders the main application layout when authenticated', () => {
    // Arrange: Mock the authenticated state
    (useAuth as vi.Mock).mockReturnValue({
      isAuthenticated: true,
      user: { username: 'testuser' },
      login: vi.fn(),
      logout: vi.fn(),
    });

    // Act: Render the App component
    render(
      <MemoryRouter initialEntries={['/']}>
        <App />
      </MemoryRouter>
    );

    // Assert: Check for elements in the protected layout
    expect(screen.getByText(/Cortex/i)).toBeInTheDocument();
    expect(screen.getByText(/UI/i)).toBeInTheDocument();
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
  });

  it('redirects to login when not authenticated', () => {
    // Arrange: Mock the unauthenticated state
    (useAuth as vi.Mock).mockReturnValue({
      isAuthenticated: false,
      user: null,
      login: vi.fn(),
      logout: vi.fn(),
    });

    // Act: Render the App component
    render(
      <MemoryRouter initialEntries={['/']}>
        <App />
      </MemoryRouter>
    );

    // Assert: Check for elements in the login view
    expect(screen.getByText('Sign in to continue')).toBeInTheDocument();
  });
});
