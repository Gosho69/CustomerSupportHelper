import { LucideIcon } from "lucide-react";

interface StatsCardProps {
  icon: LucideIcon;
  iconColor?: string;
  label: string;
  value: string | number;
  className?: string;
}

export default function StatsCard({
  icon: Icon,
  iconColor,
  label,
  value,
  className = "",
}: StatsCardProps) {
  return (
    <div
      className={`bg-white rounded-lg border p-5 ${className}`}
      style={{ borderColor: "var(--border)" }}
    >
      <div className="flex items-center justify-between mb-3">
        <p
          className="text-sm font-medium"
          style={{ color: "var(--text-secondary)" }}
        >
          {label}
        </p>
        <div
          className="w-8 h-8 rounded-md flex items-center justify-center"
          style={{ background: "var(--accent-bg)" }}
        >
          <Icon className="w-4 h-4" style={{ color: "var(--accent)" }} />
        </div>
      </div>
      <p
        className="text-2xl font-semibold"
        style={{ color: "var(--text-primary)" }}
      >
        {value}
      </p>
    </div>
  );
}
