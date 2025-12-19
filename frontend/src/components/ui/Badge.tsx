interface BadgeProps {
  children: React.ReactNode;
  variant?: "blue" | "purple" | "green" | "yellow" | "red" | "gray";
  size?: "sm" | "md" | "lg";
  className?: string;
}

const variantStyles = {
  blue: "bg-blue-500/20 text-blue-400",
  purple: "bg-purple-500/20 text-purple-400",
  green: "bg-green-500/20 text-green-400",
  yellow: "bg-yellow-500/20 text-yellow-400",
  red: "bg-red-500/20 text-red-400",
  gray: "bg-gray-500/20 text-gray-400",
};

const sizeStyles = {
  sm: "px-2 py-0.5 text-xs",
  md: "px-3 py-1 text-sm",
  lg: "px-4 py-1.5 text-base",
};

export default function Badge({
  children,
  variant = "blue",
  size = "md",
  className = "",
}: BadgeProps) {
  return (
    <span
      className={`rounded-full font-medium ${variantStyles[variant]} ${sizeStyles[size]} ${className}`}
    >
      {children}
    </span>
  );
}
