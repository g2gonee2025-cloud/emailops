import { cn } from '../../lib/utils';

interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'circular' | 'rectangular';
  width?: string | number;
  height?: string | number;
  lines?: number;
}

export function Skeleton({
  className,
  variant = 'rectangular',
  width,
  height,
  lines = 1
}: SkeletonProps) {
  const baseStyles = "animate-pulse bg-white/10 rounded";

  const variantStyles = {
    text: "h-4 rounded",
    circular: "rounded-full",
    rectangular: "rounded-lg",
  };

  const style: React.CSSProperties = {};
  if (width) style.width = typeof width === 'number' ? `${width}px` : width;
  if (height) style.height = typeof height === 'number' ? `${height}px` : height;

  if (lines > 1) {
    return (
      <div className="space-y-2">
        {Array.from({ length: lines }).map((_, i) => (
          <div
            key={i}
            className={cn(baseStyles, variantStyles[variant], className)}
            style={{
              ...style,
              width: i === lines - 1 ? '75%' : style.width // Last line shorter
            }}
          />
        ))}
      </div>
    );
  }

  return (
    <div
      className={cn(baseStyles, variantStyles[variant], className)}
      style={style}
      role="status"
      aria-label="Loading..."
    />
  );
}

// Pre-built skeleton patterns
export function SkeletonCard({ className }: { className?: string }) {
  return (
    <div className={cn("p-5 rounded-xl border border-white/10 bg-white/5 space-y-4", className)}>
      <div className="flex items-center gap-4">
        <Skeleton variant="circular" width={40} height={40} />
        <div className="flex-1 space-y-2">
          <Skeleton variant="text" width="60%" height={16} />
          <Skeleton variant="text" width="40%" height={12} />
        </div>
      </div>
      <Skeleton variant="text" lines={3} />
    </div>
  );
}

export function SkeletonTable({ rows = 5 }: { rows?: number }) {
  return (
    <div className="rounded-xl border border-white/10 overflow-hidden">
      <div className="bg-white/5 p-4">
        <Skeleton variant="text" width="30%" height={14} />
      </div>
      <div className="divide-y divide-white/5">
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="p-4 flex items-center gap-4">
            <Skeleton variant="rectangular" width={24} height={24} />
            <Skeleton variant="text" width="50%" height={14} className="flex-1" />
            <Skeleton variant="text" width={80} height={14} />
          </div>
        ))}
      </div>
    </div>
  );
}

export function SkeletonKPI() {
  return (
    <div className="p-6 rounded-xl border border-white/10 bg-white/5 space-y-3">
      <Skeleton variant="text" width="50%" height={12} />
      <Skeleton variant="text" width="40%" height={32} />
      <Skeleton variant="text" width="30%" height={12} />
    </div>
  );
}
