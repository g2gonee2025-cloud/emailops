import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '../../lib/utils';

const labelVariants = cva(
  'text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70',
  {
    variants: {
      variant: {
        default: 'text-foreground',
        glass:
          'bg-white/10 backdrop-blur-md rounded-md px-3 py-1 text-foreground border border-white/20',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  },
);

export interface LabelProps
  extends React.LabelHTMLAttributes<HTMLLabelElement>,
    VariantProps<typeof labelVariants> {}

const Label = React.forwardRef<HTMLLabelElement, LabelProps>(
  ({ className, variant, ...props }, ref) => {
    return (
      <label
        className={cn(labelVariants({ variant }), className)}
        ref={ref}
        {...props}
      />
    );
  },
);
Label.displayName = 'Label';

export { Label };
