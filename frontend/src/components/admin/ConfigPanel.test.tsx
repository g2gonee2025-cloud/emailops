
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import ConfigPanel from './ConfigPanel';
import { AppConfig } from '../../schemas/admin';
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
    expect(screen.getByLabelText(/Max Pool Size/i)).toHaveValue(mockConfig.max_pool_size?.toString());
  });

  it('calls onSave with the updated data when the form is submitted', async () => {
    const handleSave = vi.fn();
    render(
        <AllTheProviders>
            <ConfigPanel config={mockConfig} onSave={handleSave} isLoading={false} />
        </AllTheProviders>
    );

    fireEvent.change(screen.getByLabelText(/Api Url/i), { target: { value: 'http://new-api.com' } });
    fireEvent.click(screen.getByText(/Save Changes/i));

    await vi.waitFor(() => {
        expect(handleSave).toHaveBeenCalledWith({
          ...mockConfig,
          api_url: 'http://new-api.com',
        });
      });
  });

  it('displays a validation error for invalid data', async () => {
    const handleSave = vi.fn();
    render(
        <AllTheProviders>
            <ConfigPanel config={mockConfig} onSave={handleSave} isLoading={false} />
        </AllTheProviders>
    );

    fireEvent.change(screen.getByLabelText(/Api Url/i), { target: { value: 'not-a-url' } });
    fireEvent.click(screen.getByText(/Save Changes/i));

    expect(await screen.findByText(/invalid url/i)).toBeInTheDocument();
    expect(handleSave).not.toHaveBeenCalled();
  });

  it('correctly updates a number field', async () => {
    const handleSave = vi.fn();
    render(
        <AllTheProviders>
            <ConfigPanel config={mockConfig} onSave={handleSave} isLoading={false} />
        </AllTheProviders>
    );

    fireEvent.change(screen.getByLabelText(/Max Pool Size/i), { target: { value: '20' } });
    fireEvent.click(screen.getByText(/Save Changes/i));

    await vi.waitFor(() => {
        expect(handleSave).toHaveBeenCalledWith({
          ...mockConfig,
          max_pool_size: 20,
        });
      });
  });

  it('toggles visibility for sensitive fields', () => {
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
        fireEvent.click(revealButton);
    }
    expect(apiKeyInput).toHaveAttribute('type', 'text');

    if (revealButton) {
        fireEvent.click(revealButton);
    }
    expect(apiKeyInput).toHaveAttribute('type', 'password');
    });
});
