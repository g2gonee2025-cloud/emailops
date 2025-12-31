import { render, screen, waitFor } from '@testing-library/react';
import App from '../App';
import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

const routerFuture = {
    v7_startTransition: true,
    v7_relativeSplatPath: true,
};

// Mocking Recharts to avoid sizing issues in JSDOM
vi.mock('recharts', () => {
    const OriginalModule = vi.importActual('recharts');
    return {
        ...OriginalModule,
        ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div style={{ width: 800, height: 800 }}>{children}</div>,
    };
});

describe('App Smoke Test', () => {
    it('renders the sidebar navigation when authenticated', async () => {
        // Mock authenticated state
        localStorage.setItem('auth_token', 'mock-token');

        const queryClient = new QueryClient({
            defaultOptions: {
                queries: { retry: false },
            },
        });

        render(
            <QueryClientProvider client={queryClient}>
                <BrowserRouter future={routerFuture}>
                    <App />
                </BrowserRouter>
            </QueryClientProvider>
        );

        // Sidebar is part of Layout, which renders immediately for auth users
        // But we might need to wait for redirect or effect cycles if any

        await waitFor(() => {
             expect(screen.getAllByText(/Cortex/i).length).toBeGreaterThan(0);
             expect(screen.getByRole('link', { name: /Dashboard/i })).toBeInTheDocument();
        });
    });
});
