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
  color = "#635bff",
}: TrendLineChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ee" />
        <XAxis dataKey={xAxisKey} stroke="#697386" />
        <YAxis stroke="#697386" domain={[0, 100]} />
        <Tooltip
          contentStyle={{
            backgroundColor: "#ffffff",
            border: "1px solid #e3e8ee",
            borderRadius: "8px",
            color: "#1a1f36",
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
