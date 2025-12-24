import { Building2, CheckCircle, Users } from "lucide-react";
import { StatsCard } from "@/components/ui";

interface CompaniesStatsProps {
  total: number;
  active: number;
  totalEmployees: number;
}

export default function CompaniesStats({
  total,
  active,
  totalEmployees,
}: CompaniesStatsProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <StatsCard
        icon={Building2}
        iconColor="bg-indigo-500/20 text-indigo-400"
        label="Total Companies"
        value={total}
      />
      <StatsCard
        icon={CheckCircle}
        iconColor="bg-green-500/20 text-green-400"
        label="Active Companies"
        value={active}
      />
      <StatsCard
        icon={Users}
        iconColor="bg-blue-500/20 text-blue-400"
        label="Total Employees"
        value={totalEmployees}
      />
    </div>
  );
}
