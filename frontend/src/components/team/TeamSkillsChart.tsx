import { BarChart3 } from "lucide-react";
import { SkillsRadarChart } from "@/components/charts";

interface TeamSkillsChartProps {
  data: { skill: string; value: number }[];
}

export default function TeamSkillsChart({ data }: TeamSkillsChartProps) {
  return (
    <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
      <h2 className="text-xl font-bold text-white mb-6 flex items-center">
        <BarChart3 className="w-5 h-5 mr-2 text-purple-400" />
        Team Skills Overview
      </h2>
      <SkillsRadarChart data={data} />
    </div>
  );
}
