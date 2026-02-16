import { TrendingUp, TrendingDown } from "lucide-react";

interface PerformanceHeaderProps {
  latestScore: number;
  avgScore: number;
  totalReports: number;
  scoreTrend: number;
}

export default function PerformanceHeader({
  latestScore,
  avgScore,
  totalReports,
  scoreTrend,
}: PerformanceHeaderProps) {
  return (
    <div
      className="rounded-lg p-8"
      style={{ background: "#ffffff", border: "1px solid var(--border)" }}
    >
      <h1
        className="text-3xl font-bold mb-2"
        style={{ color: "var(--text-primary)" }}
      >
        My Performance Reports
      </h1>
      <p style={{ color: "var(--text-secondary)" }} className="mb-6">
        Track your progress and identify areas for improvement
      </p>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div
          className="rounded-lg p-4"
          style={{
            background: "var(--background)",
            border: "1px solid var(--border)",
          }}
        >
          <p
            style={{ color: "var(--text-secondary)" }}
            className="text-sm mb-1"
          >
            Current Score
          </p>
          <p
            className="text-3xl font-bold"
            style={{ color: "var(--text-primary)" }}
          >
            {latestScore}
          </p>
        </div>
        <div
          className="rounded-lg p-4"
          style={{
            background: "var(--background)",
            border: "1px solid var(--border)",
          }}
        >
          <p
            style={{ color: "var(--text-secondary)" }}
            className="text-sm mb-1"
          >
            Average Score
          </p>
          <p
            className="text-3xl font-bold"
            style={{ color: "var(--text-primary)" }}
          >
            {Math.round(avgScore)}
          </p>
        </div>
        <div
          className="rounded-lg p-4"
          style={{
            background: "var(--background)",
            border: "1px solid var(--border)",
          }}
        >
          <p
            style={{ color: "var(--text-secondary)" }}
            className="text-sm mb-1"
          >
            Total Reports
          </p>
          <p
            className="text-3xl font-bold"
            style={{ color: "var(--text-primary)" }}
          >
            {totalReports}
          </p>
        </div>
        <div
          className="rounded-lg p-4"
          style={{
            background: "var(--background)",
            border: "1px solid var(--border)",
          }}
        >
          <p
            style={{ color: "var(--text-secondary)" }}
            className="text-sm mb-1"
          >
            Trend
          </p>
          <div className="flex items-center space-x-2">
            <p
              className="text-3xl font-bold"
              style={{ color: "var(--text-primary)" }}
            >
              {scoreTrend > 0 ? "+" : ""}
              {scoreTrend}
            </p>
            {scoreTrend > 0 ? (
              <TrendingUp
                className="w-6 h-6"
                style={{ color: "var(--success, #0caf60)" }}
              />
            ) : (
              <TrendingDown
                className="w-6 h-6"
                style={{ color: "var(--warning, #e68a00)" }}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
