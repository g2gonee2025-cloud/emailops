import { render, screen, act, fireEvent } from '@testing-library/react';
import { AuthProvider, useAuth } from '../contexts/AuthContext';
import { MemoryRouter, Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import { vi } from 'vitest';
import api from '../lib/api';
import { useEffect, type ReactNode } from 'react';

// Mock the api module
vi.mock('../lib/api');

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value.toString();
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
  };
})();
Object.defineProperty(window, 'localStorage', { value: localStorageMock });

// A test component to display auth state and trigger actions
const AuthStatus = () => {
  const { isAuthenticated, token, login, logout } = useAuth();
  return (
    <div>
      <p>Is Authenticated: {isAuthenticated.toString()}</p>
      <p>Token: {token || 'null'}</p>
      <button onClick={() => login('testuser', 'password')}>Log In</button>
      <button onClick={() => logout()}>Log Out</button>
    </div>
  );
};

// A component that displays the current location
const LocationDisplay = () => {
  const location = useLocation();
  return <div data-testid="location-display">{location.pathname}</div>;
};

// A protected route component for testing
const ProtectedRoute = ({ children }: { children: ReactNode }) => {
    const { isAuthenticated } = useAuth();
    const navigate = useNavigate();

    useEffect(() => {
        if (!isAuthenticated) {
            navigate('/login');
        }
    }, [isAuthenticated, navigate]);

    return isAuthenticated ? <>{children}</> : null;
};


// The test setup component
const TestApp = ({ initialEntries = ['/'] }: { initialEntries?: string[] }) => {
    return (
      <MemoryRouter initialEntries={initialEntries}>
        <AuthProvider>
          <AuthStatus />
          <Routes>
            <Route path="/login" element={<div>Login Page</div>} />
            <Route path="/" element={<div>Home Page</div>} />
            <Route path="/protected" element={<ProtectedRoute><div>Protected Content</div></ProtectedRoute>} />
          </Routes>
          <LocationDisplay />
        </AuthProvider>
      </MemoryRouter>
    );
};


describe('AuthContext', () => {
  beforeEach(() => {
    localStorageMock.clear();
    // Reset mocks before each test
    vi.resetAllMocks();
  });

  it('initial state is not authenticated', () => {
    render(<TestApp />);
    expect(screen.getByText('Is Authenticated: false')).toBeInTheDocument();
    expect(screen.getByText('Token: null')).toBeInTheDocument();
  });

  it('login successfully updates the context and localStorage', async () => {
    const mockApi = api as vi.Mocked<typeof api>;
    mockApi.login.mockResolvedValue({ access_token: 'fake_token' });

    render(<TestApp />);

    await act(async () => {
      fireEvent.click(screen.getByText('Log In'));
    });

    expect(screen.getByText('Is Authenticated: true')).toBeInTheDocument();
    expect(screen.getByText('Token: fake_token')).toBeInTheDocument();
    expect(localStorage.getItem('auth_token')).toBe('fake_token');
    expect(api.setAuthToken).toHaveBeenCalledWith('fake_token');
  });

  it('logout successfully clears the context and localStorage', async () => {
    // Setup initial logged-in state
    localStorage.setItem('auth_token', 'fake_token');
    const mockApi = api as vi.Mocked<typeof api>;
    mockApi.login.mockResolvedValue({ access_token: 'fake_token' });


    render(<TestApp />);

    // Log in first to set the state
    await act(async () => {
      fireEvent.click(screen.getByText('Log In'));
    });

    expect(screen.getByText('Is Authenticated: true')).toBeInTheDocument();

    // Now log out
    await act(async () => {
        fireEvent.click(screen.getByText('Log Out'));
    });

    expect(screen.getByText('Is Authenticated: false')).toBeInTheDocument();
    expect(screen.getByText('Token: null')).toBeInTheDocument();
    expect(localStorage.getItem('auth_token')).toBeNull();
    expect(api.setAuthToken).toHaveBeenCalledWith(null);
  });

  it('handles unauthorized event by logging out and navigating to login', async () => {
    // Setup initial logged-in state
    localStorage.setItem('auth_token', 'initial_token');

    render(<TestApp initialEntries={['/protected']}/>);

    // Check initial state, should be authenticated from localStorage
    expect(await screen.findByText('Is Authenticated: true')).toBeInTheDocument();
    expect(screen.getByText('Protected Content')).toBeInTheDocument();
    expect(screen.getByTestId('location-display').textContent).toBe('/protected');

    // Dispatch the unauthorized event
    act(() => {
        window.dispatchEvent(new Event('unauthorized'));
    });

    // Check that the user is logged out
    expect(await screen.findByText('Is Authenticated: false')).toBeInTheDocument();
    expect(screen.getByText('Token: null')).toBeInTheDocument();
    expect(localStorage.getItem('auth_token')).toBeNull();

    // Check for navigation to login page
    expect(screen.getByText('Login Page')).toBeInTheDocument();
    expect(screen.getByTestId('location-display').textContent).toBe('/login');
  });
});
