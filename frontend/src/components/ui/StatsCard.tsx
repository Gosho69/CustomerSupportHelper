import { LucideIcon } from "lucide-react";

interface StatsCardProps {
  icon: LucideIcon;
  iconColor: string;
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
      className={`bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6 ${className}`}
    >
      <div
        className={`w-10 h-10 ${iconColor} rounded-lg flex items-center justify-center mb-4`}
      >
        <Icon className="w-5 h-5" />
      </div>
      <p className="text-gray-400 text-sm mb-1">{label}</p>
      <p className="text-3xl font-bold text-white">{value}</p>
    </div>
  );
}
