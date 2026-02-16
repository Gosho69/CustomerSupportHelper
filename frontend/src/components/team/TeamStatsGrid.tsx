import { Users, Phone, Award } from "lucide-react";
import { StatsCard } from "@/components/ui";

interface TeamStatsGridProps {
  totalMembers: number;
  avgPerformance: number;
  totalCalls: number;
  activeMembers: number;
}

export default function TeamStatsGrid({
  totalMembers,
  avgPerformance,
  totalCalls,
  activeMembers,
}: TeamStatsGridProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
      <StatsCard
        icon={Users}
        iconColor="text-[var(--accent)]"
        label="Total Members"
        value={totalMembers}
      />
      <StatsCard
        icon={Award}
        iconColor="text-[var(--success,#0caf60)]"
        label="Avg Performance"
        value={`${avgPerformance}%`}
      />
      <StatsCard
        icon={Phone}
        iconColor="text-[var(--accent)]"
        label="Total Calls"
        value={totalCalls}
      />
      <StatsCard
        icon={Users}
        iconColor="text-[var(--accent)]"
        label="Active Now"
        value={activeMembers}
      />
    </div>
  );
}
