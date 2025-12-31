import { render, screen, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { vi, describe, test, expect, beforeEach, type Mock } from 'vitest';
import { Sidebar } from './Sidebar';
import { useAuth } from '../contexts/AuthContext';
import { ToastProvider } from '../contexts/toastContext';

const routerFuture = {
  v7_startTransition: true,
  v7_relativeSplatPath: true,
};

// Mock the useAuth hook
vi.mock('../contexts/AuthContext');

describe('Sidebar', () => {
  const mockLogout = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    (useAuth as Mock).mockReturnValue({
      isAuthenticated: true,
      token: 'mock-token',
      user: { username: 'testuser' },
      logout: mockLogout,
      login: vi.fn(),
    });
  });

  const renderSidebar = () => {
    return render(
      <MemoryRouter future={routerFuture}>
        <ToastProvider>
          <Sidebar />
        </ToastProvider>
      </MemoryRouter>
    );
  };

  test('renders all navigation items', () => {
    renderSidebar();
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Ask')).toBeInTheDocument();
    expect(screen.getByText('Search')).toBeInTheDocument();
    expect(screen.getByText('Draft')).toBeInTheDocument();
    expect(screen.getByText('Ingestion')).toBeInTheDocument();
    expect(screen.getByText('Admin')).toBeInTheDocument();
  });

  test('collapses and expands when the toggle button is clicked', () => {
    renderSidebar();
    const collapseButton = screen.getByLabelText('Collapse sidebar');

    // Initially expanded
    expect(screen.getByText('Dashboard')).toBeVisible();

    // Collapse the sidebar
    fireEvent.click(collapseButton);
    expect(screen.queryByText('Dashboard')).toBeNull();
    expect(collapseButton).toHaveAttribute('aria-label', 'Expand sidebar');

    // Expand the sidebar
    fireEvent.click(collapseButton);
    expect(screen.getByText('Dashboard')).toBeVisible();
    expect(collapseButton).toHaveAttribute('aria-label', 'Collapse sidebar');
  });

  test('calls logout when the logout button is clicked', () => {
    renderSidebar();
    const logoutButton = screen.getByText('Logout').closest('button');
    if (logoutButton) {
      fireEvent.click(logoutButton);
    }
    expect(mockLogout).toHaveBeenCalled();
  });
});
