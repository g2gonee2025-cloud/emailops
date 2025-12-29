import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { AuthContext } from '../contexts/AuthContext';

// Mock the AuthContext
const mockAuthContext = {
  isAuthenticated: true,
  token: 'test-token',
  login: vi.fn(),
  logout: vi.fn(),
};

const renderWithProviders = (ui: React.ReactElement) => {
  return render(
    <BrowserRouter>
      <AuthContext.Provider value={mockAuthContext}>{ui}</AuthContext.Provider>
    </BrowserRouter>,
  );
};

describe('Sidebar', () => {
  it('renders all navigation links', () => {
    renderWithProviders(<Sidebar />);
    expect(screen.getByLabelText('Dashboard')).toBeInTheDocument();
    expect(screen.getByLabelText('Ask')).toBeInTheDocument();
    expect(screen.getByLabelText('Search')).toBeInTheDocument();
    expect(screen.getByLabelText('Draft')).toBeInTheDocument();
    expect(screen.getByLabelText('Summarize')).toBeInTheDocument();
    expect(screen.getByLabelText('Ingestion')).toBeInTheDocument();
    expect(screen.getByLabelText('Admin')).toBeInTheDocument();
  });

  it('navigates to the correct route on link click', () => {
    renderWithProviders(<Sidebar />);
    const dashboardLink = screen.getByLabelText('Dashboard');
    fireEvent.click(dashboardLink);
    expect(window.location.pathname).toBe('/dashboard');
  });

  it('collapses and expands the sidebar', () => {
    renderWithProviders(<Sidebar />);
    const collapseButton = screen.getByLabelText('Collapse sidebar');
    expect(screen.getByText('Collapse')).toBeInTheDocument();

    fireEvent.click(collapseButton);
    expect(screen.queryByText('Collapse')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Expand sidebar')).toBeInTheDocument();

    const expandButton = screen.getByLabelText('Expand sidebar');
    fireEvent.click(expandButton);
    expect(screen.getByText('Collapse')).toBeInTheDocument();
  });

  it('calls logout when the logout button is clicked', () => {
    renderWithProviders(<Sidebar />);
    const logoutButton = screen.getByRole('button', { name: /logout/i });
    fireEvent.click(logoutButton);
    expect(mockAuthContext.logout).toHaveBeenCalled();
  });
});
