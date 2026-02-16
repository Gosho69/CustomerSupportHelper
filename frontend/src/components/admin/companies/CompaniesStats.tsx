import { Building2, Users } from "lucide-react";
import { StatsCard } from "@/components/ui";

interface CompaniesStatsProps {
  total: number;
  totalEmployees: number;
}

export default function CompaniesStats({
  total,
  totalEmployees,
}: CompaniesStatsProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <StatsCard
        icon={Building2}
        iconColor="bg-[var(--accent-bg)] text-[var(--accent)]"
        label="Total Companies"
        value={total}
      />
      <StatsCard
        icon={Users}
        iconColor="bg-[var(--accent-bg)] text-[var(--accent)]"
        label="Total Employees"
        value={totalEmployees}
      />
    </div>
  );
}
