import { test, expect } from '@playwright/test';

test.describe('DashboardView', () => {
  test('should render dashboard components after login', async ({ page }) => {
    // Bypass login UI by setting auth token before app initialization
    await page.addInitScript(() => {
      window.localStorage.setItem('auth_token', 'dummy-e2e-token');
    });

    // Navigate to the root, which should redirect to the dashboard
    await page.goto('/');

    // Wait for the URL to change to /dashboard
    await page.waitForURL('/dashboard', { timeout: 10000 });

    // Wait for the dashboard view to be visible
    await expect(page.getByTestId('dashboard-view')).toBeVisible({ timeout: 10000 });

    // Verify header is visible
    await expect(page.getByTestId('dashboard-header')).toBeVisible();
    await expect(page.getByText('Mission Control')).toBeVisible();

    // Verify health status card is visible
    await expect(page.getByTestId('health-status-card')).toBeVisible();

    // Verify all KPI cards are visible
    await expect(page.getByTestId('kpi-card-total-emails')).toBeVisible();
    await expect(page.getByTestId('kpi-card-avg-response-time')).toBeVisible();
    await expect(page.getByTestId('kpi-card-open-rate')).toBeVisible();
    await expect(page.getByTestId('kpi-card-click-through-rate')).toBeVisible();

    // Verify main grid cards are visible
    await expect(page.getByTestId('live-stream-card')).toBeVisible();
    await expect(page.getByTestId('system-uptime-card')).toBeVisible();
    await expect(page.getByTestId('email-processing-card')).toBeVisible();
    await expect(page.getByTestId('rag-performance-card')).toBeVisible();
  });
});
