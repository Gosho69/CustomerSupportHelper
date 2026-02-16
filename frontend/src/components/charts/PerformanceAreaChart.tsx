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
  color = "#635bff",
}: PerformanceAreaChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ee" />
        <XAxis dataKey="month" stroke="#697386" />
        <YAxis stroke="#697386" domain={[0, 100]} />
        <Tooltip
          contentStyle={{
            backgroundColor: "#ffffff",
            border: "1px solid #e3e8ee",
            borderRadius: "8px",
            color: "#1a1f36",
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
