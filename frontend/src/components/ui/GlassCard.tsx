

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
      {children}
    </div>
  );
};

export default GlassCard;
