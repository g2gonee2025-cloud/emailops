import { render, screen, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { vi, describe, test, expect, beforeEach } from 'vitest';
import { Sidebar } from './Sidebar';
import { useAuth } from '../contexts/AuthContext';

// Mock the useAuth hook
vi.mock('../contexts/AuthContext');

const mockLogout = vi.fn();

describe('Sidebar', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (useAuth as vi.Mock).mockReturnValue({
      isAuthenticated: true,
      logout: mockLogout,
    });
  });

  const renderSidebar = () => {
    return render(
      <MemoryRouter>
        <Sidebar />
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
