"use client";

import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
} from "recharts";

interface SkillsRadarChartProps {
  data: { skill: string; value: number }[];
  height?: number;
  color?: string;
}

export default function SkillsRadarChart({
  data,
  height = 300,
  color = "#a855f7",
}: SkillsRadarChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <RadarChart data={data}>
        <PolarGrid stroke="#ffffff20" />
        <PolarAngleAxis dataKey="skill" stroke="#9ca3af" />
        <PolarRadiusAxis angle={90} domain={[0, 100]} stroke="#9ca3af" />
        <Radar
          name="Skills"
          dataKey="value"
          stroke={color}
          fill={color}
          fillOpacity={0.6}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
}
