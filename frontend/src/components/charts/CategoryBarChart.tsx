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
  color = "#635bff",
  layout = "vertical",
}: CategoryBarChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout={layout}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ee" />
        {layout === "vertical" ? (
          <>
            <XAxis type="number" domain={[0, 100]} stroke="#697386" />
            <YAxis
              dataKey={categoryKey}
              type="category"
              stroke="#697386"
              width={120}
            />
          </>
        ) : (
          <>
            <XAxis dataKey={categoryKey} stroke="#697386" />
            <YAxis type="number" domain={[0, 100]} stroke="#697386" />
          </>
        )}
        <Tooltip
          contentStyle={{
            backgroundColor: "#ffffff",
            border: "1px solid #e3e8ee",
            borderRadius: "8px",
            color: "#1a1f36",
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
