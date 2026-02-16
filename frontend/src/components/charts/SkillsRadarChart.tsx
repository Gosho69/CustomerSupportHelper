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
  color = "#635bff",
}: SkillsRadarChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <RadarChart data={data}>
        <PolarGrid stroke="#e3e8ee" />
        <PolarAngleAxis dataKey="skill" stroke="#697386" />
        <PolarRadiusAxis angle={90} domain={[0, 100]} stroke="#e3e8ee" />
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
