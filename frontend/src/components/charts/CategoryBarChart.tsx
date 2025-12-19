"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface CategoryBarChartProps {
  data: any[];
  dataKey: string;
  categoryKey: string;
  height?: number;
  color?: string;
  layout?: "horizontal" | "vertical";
}

export default function CategoryBarChart({
  data,
  dataKey,
  categoryKey,
  height = 250,
  color = "#8b5cf6",
  layout = "vertical",
}: CategoryBarChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout={layout}>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        {layout === "vertical" ? (
          <>
            <XAxis type="number" domain={[0, 100]} stroke="#94a3b8" />
            <YAxis
              dataKey={categoryKey}
              type="category"
              stroke="#94a3b8"
              width={120}
            />
          </>
        ) : (
          <>
            <XAxis dataKey={categoryKey} stroke="#94a3b8" />
            <YAxis type="number" domain={[0, 100]} stroke="#94a3b8" />
          </>
        )}
        <Tooltip
          contentStyle={{
            backgroundColor: "#1e293b",
            border: "1px solid #334155",
            borderRadius: "8px",
          }}
        />
        <Bar
          dataKey={dataKey}
          fill={color}
          radius={layout === "vertical" ? [0, 8, 8, 0] : [8, 8, 0, 0]}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}
