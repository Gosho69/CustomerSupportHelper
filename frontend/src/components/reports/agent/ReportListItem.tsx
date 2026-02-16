import {
  TrendingUp,
  TrendingDown,
  Calendar,
  Award,
  Target,
  Eye,
} from "lucide-react";

export interface Report {
  id: number;
  type: "weekly" | "monthly";
  period: string;
  date: string;
  score: number;
  trend: "up" | "down" | "stable";
  totalCalls: number;
  avgDuration: number;
  strengths: string[];
  improvements: string[];
  topSkills?: { skill: string; score: number }[];
}

interface ReportListItemProps {
  report: Report;
  onViewDetails: (report: Report) => void;
}

export default function ReportListItem({
  report,
  onViewDetails,
}: ReportListItemProps) {
  const getTypeColor = (type: string) => {
    switch (type) {
      case "weekly":
        return "";
      case "monthly":
        return "";
      case "quarterly":
        return "";
      default:
        return "";
    }
  };

  const getTypeStyle = (type: string) => {
    switch (type) {
      case "weekly":
        return { background: "var(--accent-bg)", color: "var(--accent)" };
      case "monthly":
        return { background: "var(--accent-bg)", color: "var(--accent)" };
      case "quarterly":
        return { background: "var(--accent-bg)", color: "var(--accent)" };
      default:
        return {
          background: "var(--accent-bg)",
          color: "var(--text-secondary)",
        };
    }
  };

  const getScoreColor = (score: number) => {
    return "";
  };

  const getScoreStyle = (score: number) => {
    if (score >= 85) return { color: "var(--success, #0caf60)" };
    if (score >= 70) return { color: "var(--warning, #e68a00)" };
    return { color: "#e25950" };
  };

  return (
    <div
      className="rounded-lg p-6 transition-all"
      style={{ background: "#ffffff", border: "1px solid var(--border)" }}
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <div className="flex items-center space-x-3 mb-2">
            <span
              className={`px-3 py-1 rounded-full text-xs font-semibold ${getTypeColor(
                report.type,
              )}`}
              style={getTypeStyle(report.type)}
            >
              {report.type.toUpperCase()}
            </span>
            <h3
              className="text-xl font-bold"
              style={{ color: "var(--text-primary)" }}
            >
              {report.period}
            </h3>
          </div>
          <p
            className="flex items-center"
            style={{ color: "var(--text-secondary)" }}
          >
            <Calendar className="w-4 h-4 mr-2" />
            {report.date}
          </p>
        </div>
        <div className="text-right">
          <div className="flex items-center justify-end space-x-2 mb-2">
            <span
              className={`text-3xl font-bold ${getScoreColor(report.score)}`}
              style={getScoreStyle(report.score)}
            >
              {report.score}
            </span>
            {report.trend === "up" ? (
              <TrendingUp
                className="w-6 h-6"
                style={{ color: "var(--success, #0caf60)" }}
              />
            ) : report.trend === "down" ? (
              <TrendingDown className="w-6 h-6" style={{ color: "#e25950" }} />
            ) : (
              <div
                className="w-6 h-1 rounded"
                style={{ background: "var(--text-secondary)" }}
              />
            )}
          </div>
          <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
            Performance Score
          </p>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div
          className="rounded-lg p-4"
          style={{
            background: "var(--background)",
            border: "1px solid var(--border)",
          }}
        >
          <p
            className="text-sm mb-1"
            style={{ color: "var(--text-secondary)" }}
          >
            Total Calls
          </p>
          <p
            className="text-2xl font-bold"
            style={{ color: "var(--text-primary)" }}
          >
            {report.totalCalls}
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
            className="text-sm mb-1"
            style={{ color: "var(--text-secondary)" }}
          >
            Avg Duration
          </p>
          <p
            className="text-2xl font-bold"
            style={{ color: "var(--text-primary)" }}
          >
            {Math.floor(report.avgDuration / 60)}:
            {String(report.avgDuration % 60).padStart(2, "0")}
          </p>
        </div>
      </div>

      {/* Strengths and Improvements */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div>
          <h4
            className="font-semibold mb-2 flex items-center"
            style={{ color: "var(--text-primary)" }}
          >
            <Award
              className="w-4 h-4 mr-2"
              style={{ color: "var(--success, #0caf60)" }}
            />
            Strengths
          </h4>
          <ul className="space-y-1">
            {report.strengths.map((strength, idx) => (
              <li
                key={idx}
                className="text-sm flex items-center"
                style={{ color: "var(--text-secondary)" }}
              >
                <span
                  className="w-1.5 h-1.5 rounded-full mr-2"
                  style={{ background: "var(--success, #0caf60)" }}
                />
                {strength}
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h4
            className="font-semibold mb-2 flex items-center"
            style={{ color: "var(--text-primary)" }}
          >
            <Target
              className="w-4 h-4 mr-2"
              style={{ color: "var(--warning, #e68a00)" }}
            />
            Areas for Improvement
          </h4>
          <ul className="space-y-1">
            {report.improvements.map((improvement, idx) => (
              <li
                key={idx}
                className="text-sm flex items-center"
                style={{ color: "var(--text-secondary)" }}
              >
                <span
                  className="w-1.5 h-1.5 rounded-full mr-2"
                  style={{ background: "var(--warning, #e68a00)" }}
                />
                {improvement}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Actions */}
      <div
        className="flex items-center space-x-3 pt-4"
        style={{ borderTop: "1px solid var(--border)" }}
      >
        <button
          onClick={() => onViewDetails(report)}
          className="flex items-center space-x-2 px-4 py-2 text-white rounded-lg transition-colors"
          style={{ background: "var(--accent)" }}
        >
          <Eye className="w-4 h-4" />
          <span>View Details</span>
        </button>
      </div>
    </div>
  );
}
