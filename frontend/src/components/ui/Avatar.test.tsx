import { render, screen } from '@/tests/testUtils';
import { vi, afterEach } from 'vitest';
import { Avatar, AvatarFallback, AvatarImage } from './Avatar';

describe('Avatar', () => {
  // Mock image loading for each test
  beforeEach(() => {
    let mockStatus: 'load' | 'error' | null = null;

    vi.spyOn(window.HTMLImageElement.prototype, 'src', 'set').mockImplementation(
      function (this: HTMLImageElement, src: string) {
        if (src === 'invalid-src') {
          mockStatus = 'error';
        } else if (src) {
          mockStatus = 'load';
        }
      },
    );

    // This is needed because Radix UI checks naturalWidth to determine if the image has loaded
    vi.spyOn(
      window.HTMLImageElement.prototype,
      'naturalWidth',
      'get',
    ).mockImplementation(() => {
      if (mockStatus === 'load') {
        // Trigger the onload event and return a width
        setTimeout(() => this.onload?.(new Event('load')), 0);
        return 150;
      }
      // Trigger the onerror event and return 0
      setTimeout(() => this.onerror?.(new Event('error')), 0);
      return 0;
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders the fallback when no image is provided', () => {
    render(
      <Avatar>
        <AvatarFallback>AV</AvatarFallback>
      </Avatar>,
    );
    expect(screen.getByText('AV')).toBeInTheDocument();
  });

  it('renders the image when a valid src is provided', async () => {
    render(
      <Avatar>
        <AvatarImage src="https://via.placeholder.com/150" alt="Avatar" />
        <AvatarFallback>AV</AvatarFallback>
      </Avatar>,
    );

    const image = await screen.findByAltText('Avatar');
    expect(image).toBeInTheDocument();
    expect(image).toHaveAttribute('src', 'https://via.placeholder.com/150');
  });

  it('renders the fallback if the image fails to load', async () => {
    render(
      <Avatar>
        <AvatarImage src="invalid-src" alt="Avatar" />
        <AvatarFallback>AV</AvatarFallback>
      </Avatar>,
    );

    const fallback = await screen.findByText('AV');
    expect(fallback).toBeInTheDocument();
    expect(screen.queryByAltText('Avatar')).not.toBeInTheDocument();
  });
});