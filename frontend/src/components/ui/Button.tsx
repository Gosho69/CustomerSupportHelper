import { LucideIcon } from "lucide-react";

interface ButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "outline" | "danger";
  size?: "sm" | "md" | "lg";
  icon?: LucideIcon;
  iconPosition?: "left" | "right";
  disabled?: boolean;
  className?: string;
  type?: "button" | "submit" | "reset";
}

const sizeStyles = {
  sm: "px-3 py-1.5 text-sm",
  md: "px-4 py-2 text-sm",
  lg: "px-5 py-2.5 text-base",
};

function getVariantStyles(variant: string): React.CSSProperties {
  switch (variant) {
    case "primary":
      return { background: "var(--accent)", color: "white" };
    case "secondary":
      return {
        background: "var(--background)",
        color: "var(--text-primary)",
        border: "1px solid var(--border)",
      };
    case "outline":
      return {
        background: "white",
        color: "var(--text-primary)",
        border: "1px solid var(--border)",
      };
    case "danger":
      return { background: "var(--danger)", color: "white" };
    default:
      return {};
  }
}

export default function Button({
  children,
  onClick,
  variant = "primary",
  size = "md",
  icon: Icon,
  iconPosition = "left",
  disabled = false,
  className = "",
  type = "button",
}: ButtonProps) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      style={getVariantStyles(variant)}
      className={`rounded-md font-medium transition-all flex items-center justify-center space-x-2 ${sizeStyles[size]} ${
        disabled ? "opacity-50 cursor-not-allowed" : "hover:opacity-90"
      } ${className}`}
    >
      {Icon && iconPosition === "left" && <Icon className="w-4 h-4" />}
      <span>{children}</span>
      {Icon && iconPosition === "right" && <Icon className="w-4 h-4" />}
    </button>
  );
}
