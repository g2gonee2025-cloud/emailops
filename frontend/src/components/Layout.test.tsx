import { render, screen } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import Layout from './Layout';

describe('Layout', () => {
  it('renders the sidebar, main content landmark, and skip link', () => {
    render(
      <MemoryRouter>
        <Layout />
      </MemoryRouter>
    );

    // Check for the sidebar
    expect(screen.getByRole('complementary')).toBeInTheDocument();
    expect(screen.getByText('Sidebar')).toBeInTheDocument();

    // Check for the main content landmark
    const main = screen.getByRole('main');
    expect(main).toBeInTheDocument();
    expect(main).toHaveAttribute('id', 'main-content');

    // Check for the skip link
    expect(screen.getByText('Skip to main content')).toBeInTheDocument();
  });

  it('renders the Outlet content', () => {
    const TestComponent = () => <div>Outlet Content</div>;
    render(
      <MemoryRouter initialEntries={['/']}>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<TestComponent />} />
          </Route>
        </Routes>
      </MemoryRouter>
    );

    expect(screen.getByText('Outlet Content')).toBeInTheDocument();
  });
});
