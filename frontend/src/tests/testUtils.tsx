
import { render as rtlRender, RenderOptions } from '@testing-library/react';
import React, { ReactElement } from 'react';

// A custom render function that can be expanded with providers
function render(ui: ReactElement, options?: Omit<RenderOptions, 'wrapper'>) {
  // You can wrap UI with providers here if needed, e.g., ThemeProvider, Router
  return rtlRender(ui, { ...options });
}

// Re-export everything from testing-library
export * from '@testing-library/react';
// Override render method with our custom one
export { render };
