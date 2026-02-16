interface BadgeProps {
  children: React.ReactNode;
  variant?: "blue" | "purple" | "green" | "yellow" | "red" | "gray";
  size?: "sm" | "md" | "lg";
  className?: string;
}

const variantStyles: Record<string, React.CSSProperties> = {
  blue: { background: "#dbeafe", color: "#1d4ed8" },
  purple: { background: "var(--accent-bg)", color: "var(--accent)" },
  green: { background: "var(--success-bg)", color: "var(--success)" },
  yellow: { background: "var(--warning-bg)", color: "var(--warning)" },
  red: { background: "var(--danger-bg)", color: "var(--danger)" },
  gray: { background: "#f0f2f5", color: "var(--text-secondary)" },
};

const sizeStyles = {
  sm: "px-2 py-0.5 text-xs",
  md: "px-2.5 py-0.5 text-xs",
  lg: "px-3 py-1 text-sm",
};

export default function Badge({
  children,
  variant = "blue",
  size = "md",
  className = "",
}: BadgeProps) {
  return (
    <span
      className={`rounded-full font-medium ${sizeStyles[size]} ${className}`}
      style={variantStyles[variant]}
    >
      {children}
    </span>
  );
}
