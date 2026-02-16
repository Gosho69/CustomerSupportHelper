import {
  Phone,
  Mail,
  Calendar,
  Award,
  TrendingUp,
  TrendingDown,
} from "lucide-react";
import { Modal, Badge, StatusIndicator } from "@/components/ui";
import { CategoryBarChart } from "@/components/charts";

interface Report {
  id: number;
  agentName: string;
  agentEmail: string;
  type: "weekly" | "monthly";
  period: string;
  date: string;
  overallScore: number;
  totalCalls: number;
  avgCallDuration: string;
  trend: "up" | "down" | "stable";
  strengths: string[];
  improvements: string[];
  topSkills: { skill: string; score: number }[];
}

interface ReportDetailModalProps {
  report: Report | null;
  onClose: () => void;
}

export default function ReportDetailModal({
  report,
  onClose,
}: ReportDetailModalProps) {
  if (!report) return null;

  const getScoreColor = (score: number) => {
    return "";
  };

  const getTrendIcon = (trend: "up" | "down" | "stable") => {
    switch (trend) {
      case "up":
        return (
          <TrendingUp
            className="w-4 h-4"
            style={{ color: "var(--success, #0caf60)" }}
          />
        );
      case "down":
        return (
          <TrendingDown
            className="w-4 h-4"
            style={{ color: "var(--warning, #e68a00)" }}
          />
        );
      default:
        return (
          <div
            className="w-4 h-4 rounded-full"
            style={{ background: "var(--text-secondary)" }}
          />
        );
    }
  };

  return (
    <Modal isOpen={!!report} onClose={onClose} size="5xl">
      {/* Header */}
      <div className="flex items-start space-x-6 mb-8">
        <div
          className="w-20 h-20 rounded-full flex items-center justify-center"
          style={{ background: "var(--accent-bg)" }}
        >
          <span
            style={{ color: "var(--accent)" }}
            className="font-bold text-3xl"
          >
            {report.agentName.charAt(0)}
          </span>
        </div>
        <div className="flex-1">
          <div className="flex items-center space-x-3 mb-2">
            <h2
              className="text-2xl font-bold"
              style={{ color: "var(--text-primary)" }}
            >
              {report.agentName}
            </h2>
            <Badge variant="gray">
              {report.type.charAt(0).toUpperCase() + report.type.slice(1)}{" "}
              Report
            </Badge>
          </div>
          <p style={{ color: "var(--text-secondary)" }}>{report.agentEmail}</p>
          <p
            className="text-sm mt-1"
            style={{ color: "var(--text-secondary)" }}
          >
            Period: {report.period}
          </p>
        </div>
        <div className="text-right">
          <p
            className="text-sm mb-1"
            style={{ color: "var(--text-secondary)" }}
          >
            Overall Score
          </p>
          <p
            className={`text-4xl font-bold ${getScoreColor(
              report.overallScore,
            )}`}
            style={{ color: "var(--text-primary)" }}
          >
            {report.overallScore}%
          </p>
          {getTrendIcon(report.trend)}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 mb-8">
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
            {report.avgCallDuration}
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
            Trend
          </p>
          <p
            className="text-xl font-bold capitalize"
            style={{ color: "var(--text-primary)" }}
          >
            {report.trend}
          </p>
        </div>
      </div>

      {/* Skills Chart */}
      <div
        className="rounded-lg p-6 mb-8"
        style={{
          background: "var(--background)",
          border: "1px solid var(--border)",
        }}
      >
        <h3
          className="text-lg font-semibold mb-4"
          style={{ color: "var(--text-primary)" }}
        >
          Skills Breakdown
        </h3>
        <CategoryBarChart
          data={report.topSkills}
          dataKey="score"
          categoryKey="skill"
          layout="horizontal"
        />
      </div>

      {/* Strengths and Improvements */}
      <div className="grid grid-cols-2 gap-6 mb-8">
        <div
          className="rounded-lg p-6"
          style={{
            background: "var(--background)",
            border: "1px solid var(--border)",
          }}
        >
          <h3
            className="text-lg font-semibold mb-4 flex items-center"
            style={{ color: "var(--text-primary)" }}
          >
            <Award
              className="w-5 h-5 mr-2"
              style={{ color: "var(--success, #0caf60)" }}
            />
            Strengths
          </h3>
          <ul className="space-y-2">
            {report.strengths.map((strength, index) => (
              <li
                key={index}
                className="flex items-start text-sm"
                style={{ color: "var(--text-secondary)" }}
              >
                <span
                  style={{ color: "var(--success, #0caf60)" }}
                  className="mr-2"
                >
                  ✓
                </span>
                {strength}
              </li>
            ))}
          </ul>
        </div>

        <div
          className="rounded-lg p-6"
          style={{
            background: "var(--background)",
            border: "1px solid var(--border)",
          }}
        >
          <h3
            className="text-lg font-semibold mb-4 flex items-center"
            style={{ color: "var(--text-primary)" }}
          >
            <TrendingUp
              className="w-5 h-5 mr-2"
              style={{ color: "var(--accent)" }}
            />
            Areas for Improvement
          </h3>
          <ul className="space-y-2">
            {report.improvements.map((improvement, index) => (
              <li
                key={index}
                className="flex items-start text-sm"
                style={{ color: "var(--text-secondary)" }}
              >
                <span style={{ color: "var(--accent)" }} className="mr-2">
                  →
                </span>
                {improvement}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </Modal>
  );
}
