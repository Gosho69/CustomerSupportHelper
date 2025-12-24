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
        iconColor="bg-indigo-500/20 text-indigo-400"
        label="Total Users"
        value={total}
      />
      <StatsCard
        icon={Shield}
        iconColor="bg-purple-500/20 text-purple-400"
        label="Admins"
        value={admins}
      />
      <StatsCard
        icon={UserCog}
        iconColor="bg-blue-500/20 text-blue-400"
        label="Heads"
        value={heads}
      />
      <StatsCard
        icon={Headphones}
        iconColor="bg-cyan-500/20 text-cyan-400"
        label="Agents"
        value={agents}
      />
      <StatsCard
        icon={CheckCircle}
        iconColor="bg-green-500/20 text-green-400"
        label="Active"
        value={active}
      />
    </div>
  );
}
