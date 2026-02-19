"use client";

import { Phone, Star, Clock, Target, Activity } from "lucide-react";
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

interface PerformanceStatsProps {
  stats: { totalCalls: number; avgScore: number; totalHours: number };
  skillData: any[];
  monthlyActivity: any[];
  loading?: boolean;
}

const cardStyle: React.CSSProperties = {
  background: "#ffffff",
  border: "1px solid var(--border, #e3e8ee)",
  borderRadius: "8px",
};

function Skeleton({ className = "" }: { className?: string }) {
  return (
    <div
      className={`animate-pulse rounded ${className}`}
      style={{ background: "var(--border, #e3e8ee)" }}
    />
  );
}

export default function PerformanceStats({
  stats,
  skillData,
  monthlyActivity,
  loading = false,
}: PerformanceStatsProps) {
  if (loading) {
    return (
      <>
        {/* Stat card skeletons */}
        <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
          {[0, 1, 2].map((i) => (
            <div key={i} className="rounded-lg p-6" style={cardStyle}>
              <Skeleton className="w-10 h-10 mb-4" />
              <Skeleton className="h-3 w-20 mb-2" />
              <Skeleton className="h-8 w-16" />
            </div>
          ))}
        </div>

        {/* Chart skeletons */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {[0, 1].map((i) => (
            <div key={i} className="rounded-lg p-6" style={cardStyle}>
              <div className="flex items-center gap-2 mb-4">
                <Skeleton className="w-5 h-5" />
                <Skeleton className="h-5 w-36" />
              </div>
              <Skeleton className="w-full h-[300px]" />
            </div>
          ))}
        </div>
      </>
    );
  }

  return (
    <>
      {/* Stat Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
        <div className="rounded-lg p-6" style={cardStyle}>
          <div
            className="w-10 h-10 rounded-lg flex items-center justify-center mb-4"
            style={{ background: "var(--accent-bg)" }}
          >
            <Phone className="w-5 h-5" style={{ color: "var(--accent)" }} />
          </div>
          <p
            className="text-sm mb-1"
            style={{ color: "var(--text-secondary)" }}
          >
            Total Calls
          </p>
          <p
            className="text-3xl font-bold"
            style={{ color: "var(--text-primary)" }}
          >
            {stats.totalCalls}
          </p>
        </div>

        <div className="rounded-lg p-6" style={cardStyle}>
          <div
            className="w-10 h-10 rounded-lg flex items-center justify-center mb-4"
            style={{ background: "var(--accent-bg)" }}
          >
            <Star className="w-5 h-5" style={{ color: "var(--accent)" }} />
          </div>
          <p
            className="text-sm mb-1"
            style={{ color: "var(--text-secondary)" }}
          >
            Avg Score
          </p>
          <p
            className="text-3xl font-bold"
            style={{ color: "var(--text-primary)" }}
          >
            {stats.avgScore}
          </p>
        </div>

        <div className="rounded-lg p-6" style={cardStyle}>
          <div
            className="w-10 h-10 rounded-lg flex items-center justify-center mb-4"
            style={{ background: "var(--accent-bg)" }}
          >
            <Clock className="w-5 h-5" style={{ color: "var(--accent)" }} />
          </div>
          <p
            className="text-sm mb-1"
            style={{ color: "var(--text-secondary)" }}
          >
            Total Hours
          </p>
          <p
            className="text-3xl font-bold"
            style={{ color: "var(--text-primary)" }}
          >
            {stats.totalHours}h
          </p>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Skill Radar */}
        <div className="rounded-lg p-6" style={cardStyle}>
          <h3
            className="text-xl font-bold mb-4 flex items-center"
            style={{ color: "var(--text-primary)" }}
          >
            <Target
              className="w-5 h-5 mr-2"
              style={{ color: "var(--accent)" }}
            />
            Skill Assessment
          </h3>
          {skillData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={skillData}>
                <PolarGrid stroke="#e3e8ee" />
                <PolarAngleAxis dataKey="skill" stroke="#697386" />
                <PolarRadiusAxis domain={[0, 100]} stroke="#697386" />
                <Radar
                  name="Score"
                  dataKey="score"
                  stroke="#635bff"
                  fill="#635bff"
                  fillOpacity={0.6}
                />
              </RadarChart>
            </ResponsiveContainer>
          ) : (
            <div
              className="h-[300px] flex items-center justify-center"
              style={{ color: "var(--text-secondary)" }}
            >
              No skill data available yet
            </div>
          )}
        </div>

        {/* Activity Chart */}
        <div className="rounded-lg p-6" style={cardStyle}>
          <h3
            className="text-xl font-bold mb-4 flex items-center"
            style={{ color: "var(--text-primary)" }}
          >
            <Activity
              className="w-5 h-5 mr-2"
              style={{ color: "var(--accent)" }}
            />
            Monthly Activity
          </h3>
          {monthlyActivity.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={monthlyActivity}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ee" />
                <XAxis dataKey="month" stroke="#697386" />
                <YAxis yAxisId="left" stroke="#697386" />
                <YAxis yAxisId="right" orientation="right" stroke="#697386" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#ffffff",
                    border: "1px solid #e3e8ee",
                    borderRadius: "8px",
                    color: "#1a1f36",
                  }}
                />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="calls"
                  stroke="#635bff"
                  strokeWidth={2}
                  name="Calls"
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="score"
                  stroke="#0caf60"
                  strokeWidth={2}
                  name="Score"
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div
              className="h-[300px] flex items-center justify-center"
              style={{ color: "var(--text-secondary)" }}
            >
              No activity data available yet
            </div>
          )}
        </div>
      </div>
    </>
  );
}
