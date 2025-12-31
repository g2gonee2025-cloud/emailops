import '@testing-library/jest-dom/vitest';
import { vi } from 'vitest';

// Mock scrollIntoView for Radix UI components
window.Element.prototype.scrollIntoView = vi.fn();
