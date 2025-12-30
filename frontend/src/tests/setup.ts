import '@testing-library/jest-dom';

// Mock scrollIntoView for Radix UI components
window.Element.prototype.scrollIntoView = vi.fn();
