import * as React from 'react';
import { cn } from '../../lib/utils';

interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value?: number;
  max?: number;
  size?: 'small' | 'medium' | 'large';
  variant?: 'primary' | 'secondary' | 'success' | 'warning' | 'danger';
  label?: string;
  showValue?: boolean;
}

const Progress = React.forwardRef<HTMLDivElement, ProgressProps>(
  (
    {
      className,
      value = 0,
      max = 100,
      size = 'medium',
      variant = 'primary',
      label,
      showValue = false,
      ...props
    },
    ref,
  ) => {
    const percentage = max > 0 ? (value / max) * 100 : 0;

    const sizeClasses = {
      small: 'h-1.5',
      medium: 'h-2.5',
      large: 'h-4',
    };

    const variantClasses = {
      primary: 'bg-blue-500',
      secondary: 'bg-gray-500',
      success: 'bg-green-500',
      warning: 'bg-yellow-500',
      danger: 'bg-red-500',
    };

    return (
      <div className="w-full">
        {label && (
          <div className="flex justify-between mb-1">
            <span className="text-base font-medium text-white">{label}</span>
            {showValue && (
              <span className="text-sm font-medium text-white">{`${Math.round(
                percentage,
              )}%`}</span>
            )}
          </div>
        )}
        <div
          ref={ref}
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={max}
          aria-valuenow={value}
          className={cn(
            'relative w-full overflow-hidden rounded-full bg-white/10 backdrop-blur-md border border-white/10 shadow-inner',
            sizeClasses[size],
            className,
          )}
          {...props}
        >
          <div
            className={cn(
              'h-full rounded-full transition-all duration-500 ease-out',
              variantClasses[variant],
            )}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>
    );
  },
);

Progress.displayName = 'Progress';

export { Progress };
