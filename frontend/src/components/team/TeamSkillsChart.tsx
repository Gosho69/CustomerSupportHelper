import { BarChart3 } from "lucide-react";
import { SkillsRadarChart } from "@/components/charts";

interface TeamSkillsChartProps {
  data: { skill: string; value: number }[];
}

export default function TeamSkillsChart({ data }: TeamSkillsChartProps) {
  return (
    <div
      className="rounded-lg p-6"
      style={{
        background: "#ffffff",
        border: "1px solid var(--border)",
        borderRadius: "8px",
      }}
    >
      <h2
        className="text-xl font-bold mb-6 flex items-center"
        style={{ color: "var(--text-primary)" }}
      >
        <BarChart3
          className="w-5 h-5 mr-2"
          style={{ color: "var(--accent)" }}
        />
        Team Skills Overview
      </h2>
      <SkillsRadarChart data={data} />
    </div>
  );
}
