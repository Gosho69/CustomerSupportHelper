"use client";

import { TrendingUp, Target } from "lucide-react";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface PerformanceChartsProps {
  performanceData: { month: string; score: number; calls: number }[];
  categoryScores: { category: string; score: number }[];
}

export default function PerformanceCharts({
  performanceData,
  categoryScores,
}: PerformanceChartsProps) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Score Trend */}
      <div
        className="rounded-lg p-6"
        style={{ background: "#ffffff", border: "1px solid var(--border)" }}
      >
        <h3
          className="text-xl font-bold mb-4 flex items-center"
          style={{ color: "var(--text-primary)" }}
        >
          <TrendingUp
            className="w-5 h-5 mr-2"
            style={{ color: "var(--accent)" }}
          />
          Performance Trend
        </h3>
        <ResponsiveContainer width="100%" height={250}>
          <AreaChart data={performanceData}>
            <defs>
              <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#635bff" stopOpacity={0.15} />
                <stop offset="95%" stopColor="#635bff" stopOpacity={0} />
              </linearGradient>
            </defs>
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
              stroke="#635bff"
              fillOpacity={1}
              fill="url(#colorScore)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Category Scores */}
      <div
        className="rounded-lg p-6"
        style={{ background: "#ffffff", border: "1px solid var(--border)" }}
      >
        <h3
          className="text-xl font-bold mb-4 flex items-center"
          style={{ color: "var(--text-primary)" }}
        >
          <Target className="w-5 h-5 mr-2" style={{ color: "var(--accent)" }} />
          Skill Categories
        </h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={categoryScores} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ee" />
            <XAxis type="number" domain={[0, 100]} stroke="#697386" />
            <YAxis
              dataKey="category"
              type="category"
              stroke="#697386"
              width={120}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#ffffff",
                border: "1px solid #e3e8ee",
                borderRadius: "8px",
                color: "#1a1f36",
              }}
            />
            <Bar dataKey="score" fill="#635bff" radius={[0, 8, 8, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
