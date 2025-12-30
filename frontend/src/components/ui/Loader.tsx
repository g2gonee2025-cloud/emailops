import { Loader2 } from "lucide-react"

import { cn } from "../../lib/utils"

export interface LoaderProps extends React.HTMLAttributes<HTMLDivElement> {
  size?: number | string
}

export function Loader({ className, size = 24, ...props }: LoaderProps) {
  return (
    <div
      className={cn("flex items-center justify-center text-primary", className)}
      {...props}
    >
      <Loader2 className="animate-spin" size={size} />
      <span className="sr-only">Loading...</span>
    </div>
  )
}
