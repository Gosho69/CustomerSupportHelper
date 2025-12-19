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
        iconColor="bg-blue-500/20 text-blue-400"
        label="Total Members"
        value={totalMembers}
      />
      <StatsCard
        icon={Award}
        iconColor="bg-green-500/20 text-green-400"
        label="Avg Performance"
        value={`${avgPerformance}%`}
      />
      <StatsCard
        icon={Phone}
        iconColor="bg-cyan-500/20 text-cyan-400"
        label="Total Calls"
        value={totalCalls}
      />
      <StatsCard
        icon={Users}
        iconColor="bg-purple-500/20 text-purple-400"
        label="Active Now"
        value={activeMembers}
      />
    </div>
  );
}
