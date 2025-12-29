import * as React from 'react';

import { cn } from '../../lib/utils';

const sizeStyles = {
  sm: 'h-9 px-3 py-1 text-xs',
  md: 'h-10 px-3 py-2 text-sm',
  lg: 'h-11 px-4 py-2 text-base',
};

export type InputSize = keyof typeof sizeStyles;

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  size?: InputSize;
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, size = 'md', ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          'flex w-full rounded-md border border-white/10 bg-white/5 text-white/80 ring-offset-background backdrop-blur-sm file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-white/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 transition-all duration-300 focus:border-white/20 focus:bg-white/10',
          sizeStyles[size],
          className
        )}
        ref={ref}
        {...props}
      />
    );
  }
);
Input.displayName = 'Input';

export { Input };
