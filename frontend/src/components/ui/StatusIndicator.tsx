interface StatusIndicatorProps {
  status: "active" | "on_break" | "offline" | string;
  label?: string;
  showDot?: boolean;
}

const statusColors = {
  active: "bg-green-500",
  on_break: "bg-yellow-500",
  offline: "bg-gray-500",
};

const statusLabels = {
  active: "Active",
  on_break: "On Break",
  offline: "Offline",
};

export default function StatusIndicator({
  status,
  label,
  showDot = true,
}: StatusIndicatorProps) {
  const dotColor =
    statusColors[status as keyof typeof statusColors] || "bg-gray-500";
  const displayLabel =
    label || statusLabels[status as keyof typeof statusLabels] || status;

  return (
    <div className="flex items-center space-x-2">
      {showDot && <div className={`w-2 h-2 rounded-full ${dotColor}`} />}
      <span className="text-xs text-gray-400">{displayLabel}</span>
    </div>
  );
}
