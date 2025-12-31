import { test, expect } from '@playwright/test';

test('should allow a user to log in and be redirected to the dashboard', async ({ page }) => {
  await page.route('**/api/v1/auth/login', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        access_token: 'dummy-e2e-token',
        token_type: 'bearer',
        expires_in: 3600,
      }),
    });
  });

  // Navigate to the login page
  await page.goto('/login');

  // Check that the page is the login page
  await expect(page).toHaveURL('/login');

  // Fill in the email and password fields
  await page.getByTestId('email-input').fill('testuser@emailops.ai');
  await page.getByTestId('password-input').fill('test');

  // Click the sign in button
  await page.getByTestId('submit-button').click();

  // Check that the user is redirected to the dashboard
  await expect(page).toHaveURL('/dashboard');

  // Check that the dashboard heading is visible
  await expect(page.getByTestId('dashboard-title')).toBeVisible();
});
