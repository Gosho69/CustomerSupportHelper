"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface TrendLineChartProps {
  data: any[];
  dataKey: string;
  xAxisKey?: string;
  height?: number;
  color?: string;
}

export default function TrendLineChart({
  data,
  dataKey,
  xAxisKey = "month",
  height = 300,
  color = "#a855f7",
}: TrendLineChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
        <XAxis dataKey={xAxisKey} stroke="#9ca3af" />
        <YAxis stroke="#9ca3af" domain={[0, 100]} />
        <Tooltip
          contentStyle={{
            backgroundColor: "#1e293b",
            border: "1px solid #ffffff20",
            borderRadius: "8px",
          }}
        />
        <Line
          type="monotone"
          dataKey={dataKey}
          stroke={color}
          strokeWidth={3}
          dot={{ fill: color, r: 6 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
