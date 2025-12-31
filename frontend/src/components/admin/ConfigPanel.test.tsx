
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import ConfigPanel from './ConfigPanel';
import type { AppConfig } from '../../schemas/admin';
import { AllTheProviders } from '../../tests/testUtils';


const mockConfig: AppConfig = {
  api_url: 'http://localhost:8000',
  log_level: 'INFO',
  max_pool_size: 10,
};

describe('ConfigPanel', () => {
  it('renders the form with initial values', () => {
    render(
        <AllTheProviders>
            <ConfigPanel config={mockConfig} onSave={vi.fn()} isLoading={false} />
        </AllTheProviders>
    );

    expect(screen.getByLabelText(/Api Url/i)).toHaveValue(mockConfig.api_url);
    expect(screen.getByLabelText(/Log Level/i)).toHaveValue(mockConfig.log_level);
    expect(screen.getByLabelText(/Max Pool Size/i)).toHaveValue(mockConfig.max_pool_size);
  });

  it('calls onSave with the updated data when the form is submitted', async () => {
    const handleSave = vi.fn();
    const user = userEvent.setup();
    render(
        <AllTheProviders>
            <ConfigPanel config={mockConfig} onSave={handleSave} isLoading={false} />
        </AllTheProviders>
    );

    await user.clear(screen.getByLabelText(/Api Url/i));
    await user.type(screen.getByLabelText(/Api Url/i), 'http://new-api.com');
    await user.click(screen.getByText(/Save Changes/i));

    await vi.waitFor(() => {
        expect(handleSave).toHaveBeenCalledWith({
          ...mockConfig,
          api_url: 'http://new-api.com',
        });
      });
  });

  it('displays a validation error for invalid data', async () => {
    const handleSave = vi.fn();
    const user = userEvent.setup();
    render(
        <AllTheProviders>
            <ConfigPanel config={mockConfig} onSave={handleSave} isLoading={false} />
        </AllTheProviders>
    );

    await user.clear(screen.getByLabelText(/Api Url/i));
    await user.type(screen.getByLabelText(/Api Url/i), 'not-a-url');
    await user.click(screen.getByText(/Save Changes/i));

    expect(await screen.findByText(/invalid url/i)).toBeInTheDocument();
    expect(handleSave).not.toHaveBeenCalled();
  });

  it('correctly updates a number field', async () => {
    const handleSave = vi.fn();
    const user = userEvent.setup();
    render(
        <AllTheProviders>
            <ConfigPanel config={mockConfig} onSave={handleSave} isLoading={false} />
        </AllTheProviders>
    );

    await user.clear(screen.getByLabelText(/Max Pool Size/i));
    await user.type(screen.getByLabelText(/Max Pool Size/i), '20');
    await user.click(screen.getByText(/Save Changes/i));

    await vi.waitFor(() => {
        expect(handleSave).toHaveBeenCalledWith({
          ...mockConfig,
          max_pool_size: 20,
        });
      });
  });

  it('toggles visibility for sensitive fields', async () => {
    const user = userEvent.setup();
    const sensitiveConfig = { ...mockConfig, api_key: 'supersecret' };
    render(
        <AllTheProviders>
            <ConfigPanel config={sensitiveConfig} onSave={vi.fn()} isLoading={false} />
        </AllTheProviders>
    );

    const apiKeyInput = screen.getByLabelText(/Api Key/i);
    expect(apiKeyInput).toHaveAttribute('type', 'password');

    const revealButton = apiKeyInput.nextElementSibling;
    expect(revealButton).toBeInTheDocument();

    if (revealButton) {
        await user.click(revealButton);
    }
    expect(apiKeyInput).toHaveAttribute('type', 'text');

    if (revealButton) {
        await user.click(revealButton);
    }
    expect(apiKeyInput).toHaveAttribute('type', 'password');
    });
});
