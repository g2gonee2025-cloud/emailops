import { test, expect } from '@playwright/test';

test.describe('DashboardView', () => {
  test('should render dashboard components after login', async ({ page }) => {
    // Navigate to the login page to set up local storage
    await page.goto('/login');

    // Bypass login UI by setting auth token directly in local storage
    await page.evaluate(() => {
      localStorage.setItem('auth_token', 'dummy-e2e-token');
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
    await expect(page.getByTestId('kpi-card-pipeline-throughput')).toBeVisible();
    await expect(page.getByTestId('kpi-card-active-connections')).toBeVisible();
    await expect(page.getByTestId('kpi-card-vector-index')).toBeVisible();
    await expect(page.getByTestId('kpi-card-security-gate')).toBeVisible();

    // Verify main grid cards are visible
    await expect(page.getByTestId('live-stream-card')).toBeVisible();
    await expect(page.getByTestId('system-uptime-card')).toBeVisible();
    await expect(page.getByTestId('email-processing-card')).toBeVisible();
    await expect(page.getByTestId('rag-performance-card')).toBeVisible();
  });
});
