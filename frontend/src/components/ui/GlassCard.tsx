

import { cn } from '../../lib/utils';

interface GlassCardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  intensity?: 'low' | 'medium' | 'high';
  hoverEffect?: boolean;
}

const GlassCard: React.FC<GlassCardProps> = ({
  children,
  className,
  intensity = 'medium',
  hoverEffect = false,
  ...props
}) => {
  const baseStyles = "relative overflow-hidden rounded-xl border border-white/10 shadow-xl transition-all duration-300";

  const intensities = {
    low: "bg-white/5 backdrop-blur-sm",
    medium: "bg-white/10 backdrop-blur-md",
    high: "bg-white/20 backdrop-blur-lg",
  };

  const hoverStyles = hoverEffect
    ? "hover:bg-white/15 hover:scale-[1.02] hover:shadow-2xl hover:border-white/20"
    : "";

  return (
    <div
      className={cn(baseStyles, intensities[intensity], hoverStyles, className)}
      {...props}
    >
      {/* Noise texture overlay for premium feel */}
      <div className="absolute inset-0 opacity-[0.03] pointer-events-none z-0"
           style={{ backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E")` }}
      />
      <div className="relative z-10 w-full h-full">
        {children}
      </div>
    </div>
  );
};

export default GlassCard;
