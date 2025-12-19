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

const variantStyles = {
  primary:
    "bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white",
  secondary: "bg-slate-700 hover:bg-slate-600 text-white",
  outline:
    "bg-slate-800/50 hover:bg-slate-800 border border-white/10 text-white",
  danger: "bg-red-600 hover:bg-red-700 text-white",
};

const sizeStyles = {
  sm: "px-4 py-2 text-sm",
  md: "px-6 py-3 text-base",
  lg: "px-8 py-4 text-lg",
};

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
      className={`rounded-xl font-semibold transition-all flex items-center justify-center space-x-2 ${
        variantStyles[variant]
      } ${sizeStyles[size]} ${
        disabled ? "opacity-50 cursor-not-allowed" : ""
      } ${className}`}
    >
      {Icon && iconPosition === "left" && <Icon className="w-5 h-5" />}
      <span>{children}</span>
      {Icon && iconPosition === "right" && <Icon className="w-5 h-5" />}
    </button>
  );
}
