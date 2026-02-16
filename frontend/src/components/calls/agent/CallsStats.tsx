import { Phone, Clock, TrendingUp } from "lucide-react";
import { StatsCard } from "@/components/ui";

interface CallsStatsProps {
  total: number;
  avgDuration: string;
  today: number;
}

export default function CallsStats({
  total,
  avgDuration,
  today,
}: CallsStatsProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <StatsCard
        icon={Phone}
        iconColor="bg-blue-50 text-blue-600"
        label="Total Calls"
        value={total}
      />
      <StatsCard
        icon={Clock}
        iconColor="bg-cyan-50 text-cyan-600"
        label="Avg Duration"
        value={avgDuration}
      />
      <StatsCard
        icon={TrendingUp}
        iconColor="bg-purple-50 text-purple-600"
        label="Today"
        value={today}
      />
    </div>
  );
}
