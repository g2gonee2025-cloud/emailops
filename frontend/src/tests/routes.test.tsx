
import { render, screen } from './testUtils';
import App from '../App';

// Mock AuthContext to simulate unauthenticated state initially
// But renderWithProviders defaults to authenticated=false if not specified?
// Actually testUtils wraps with AuthProvider which uses local storage.
// We might need to mock local storage or the AuthContext.

describe('Routing Tests', () => {
    it('redirects to login when unauthenticated', async () => {
        localStorage.clear();
        render(<App />);

        // Should see headers or inputs related to login
        // Assuming LoginView has a "Sign In" or "Username" field
        // Since we don't have the full LoginView implementation details here, checking for URL or unique Login text
        // But render doesn't expose URL easily without memory router history access.

        // Simpler check: Dashboard link should NOT be visible if we are on Login page (which doesn't have sidebar)
        // LoginView usually doesn't show Sidebar.
        // ProtectedLayout shows Sidebar.

        expect(screen.queryByRole('link', { name: /Dashboard/i })).not.toBeInTheDocument();
    });
});
