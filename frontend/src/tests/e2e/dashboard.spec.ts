import { test, expect } from '@playwright/test';

test.describe('DashboardView', () => {
  test('should render dashboard components after login', async ({ page }) => {
    await page.addInitScript(() => {
      window.localStorage.setItem('auth_token', 'dummy-e2e-token');
    });

    await page.goto('/');

    await page.waitForURL('/dashboard', { timeout: 10000 });

    await expect(page.getByTestId('dashboard-view')).toBeVisible({ timeout: 10000 });

    await expect(page.getByTestId('dashboard-header')).toBeVisible();
    await expect(page.getByText('Mission Control')).toBeVisible();

    await expect(page.getByTestId('health-status-card')).toBeVisible();

    await expect(page.getByText('No KPI Data Available')).toBeVisible();

    await expect(page.getByTestId('live-stream-card')).toBeVisible();
    await expect(page.getByTestId('system-uptime-card')).toBeVisible();
    await expect(page.getByTestId('email-processing-card')).toBeVisible();
    await expect(page.getByTestId('rag-performance-card')).toBeVisible();
  });
});
