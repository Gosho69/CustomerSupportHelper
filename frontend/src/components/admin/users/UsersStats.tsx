import { Users, Shield, UserCog, Headphones, CheckCircle } from "lucide-react";
import { StatsCard } from "@/components/ui";

interface UsersStatsProps {
  total: number;
  admins: number;
  heads: number;
  agents: number;
  active: number;
}

export default function UsersStats({
  total,
  admins,
  heads,
  agents,
  active,
}: UsersStatsProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
      <StatsCard
        icon={Users}
        iconColor="bg-[var(--accent-bg)] text-[var(--accent)]"
        label="Total Users"
        value={total}
      />
      <StatsCard
        icon={Shield}
        iconColor="bg-[var(--accent-bg)] text-[var(--accent)]"
        label="Admins"
        value={admins}
      />
      <StatsCard
        icon={UserCog}
        iconColor="bg-[var(--accent-bg)] text-[var(--accent)]"
        label="Heads"
        value={heads}
      />
      <StatsCard
        icon={Headphones}
        iconColor="bg-[var(--accent-bg)] text-[var(--accent)]"
        label="Agents"
        value={agents}
      />
      <StatsCard
        icon={CheckCircle}
        iconColor="bg-[var(--accent-bg)] text-[var(--accent)]"
        label="Active"
        value={active}
      />
    </div>
  );
}
