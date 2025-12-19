"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface PerformanceAreaChartProps {
  data: { month: string; score: number }[];
  height?: number;
  color?: string;
}

export default function PerformanceAreaChart({
  data,
  height = 250,
  color = "#a855f7",
}: PerformanceAreaChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
        <XAxis dataKey="month" stroke="#9ca3af" />
        <YAxis stroke="#9ca3af" domain={[0, 100]} />
        <Tooltip
          contentStyle={{
            backgroundColor: "#1e293b",
            border: "1px solid #ffffff20",
            borderRadius: "8px",
          }}
        />
        <Area
          type="monotone"
          dataKey="score"
          stroke={color}
          fill={color}
          fillOpacity={0.6}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
