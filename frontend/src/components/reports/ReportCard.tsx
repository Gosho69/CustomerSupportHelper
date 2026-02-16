import {
  Calendar,
  TrendingUp,
  TrendingDown,
  Eye,
  Download,
} from "lucide-react";
import { Badge } from "@/components/ui";

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
}

interface ReportCardProps {
  report: Report;
  onView: () => void;
  onDownload: () => void;
}

export default function ReportCard({
  report,
  onView,
  onDownload,
}: ReportCardProps) {
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
    <div
      className="rounded-lg p-5 transition-all"
      style={{
        background: "#ffffff",
        border: "1px solid var(--border)",
      }}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4 flex-1">
          <div
            className="w-12 h-12 rounded-full flex items-center justify-center"
            style={{ background: "var(--accent-bg)" }}
          >
            <span
              style={{ color: "var(--accent)" }}
              className="font-bold text-lg"
            >
              {report.agentName.charAt(0)}
            </span>
          </div>
          <div className="flex-1">
            <div className="flex items-center space-x-3 mb-1">
              <p
                className="font-semibold text-lg"
                style={{ color: "var(--text-primary)" }}
              >
                {report.agentName}
              </p>
              <Badge
                variant={report.type === "weekly" ? "blue" : "purple"}
                size="sm"
              >
                {report.type.charAt(0).toUpperCase() + report.type.slice(1)}
              </Badge>
            </div>
            <div
              className="flex items-center space-x-4 text-sm"
              style={{ color: "var(--text-secondary)" }}
            >
              <span className="flex items-center">
                <Calendar className="w-3 h-3 mr-1" />
                {report.period}
              </span>
              <span>Generated on {report.date}</span>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-6">
          <div className="text-center">
            <p
              className="text-xs mb-1"
              style={{ color: "var(--text-secondary)" }}
            >
              Score
            </p>
            <p
              className={`text-2xl font-bold ${getScoreColor(
                report.overallScore,
              )}`}
              style={{ color: "var(--text-primary)" }}
            >
              {report.overallScore}%
            </p>
          </div>
          <div className="text-center">
            <p
              className="text-xs mb-1"
              style={{ color: "var(--text-secondary)" }}
            >
              Calls
            </p>
            <p
              className="font-semibold text-lg"
              style={{ color: "var(--text-primary)" }}
            >
              {report.totalCalls}
            </p>
          </div>
          <div className="text-center">
            <p
              className="text-xs mb-1"
              style={{ color: "var(--text-secondary)" }}
            >
              Avg Duration
            </p>
            <p
              className="font-semibold"
              style={{ color: "var(--text-primary)" }}
            >
              {report.avgCallDuration}
            </p>
          </div>
          <div className="flex flex-col items-center">
            <p
              className="text-xs mb-1"
              style={{ color: "var(--text-secondary)" }}
            >
              Trend
            </p>
            {getTrendIcon(report.trend)}
          </div>
          <div className="flex space-x-2">
            <button
              onClick={onView}
              className="p-2 rounded-lg transition-colors"
              style={{ background: "var(--accent-bg)" }}
            >
              <Eye
                className="w-5 h-5"
                style={{ color: "var(--text-secondary)" }}
              />
            </button>
            <button
              onClick={onDownload}
              className="p-2 rounded-lg transition-colors"
              style={{ background: "var(--accent-bg)" }}
            >
              <Download
                className="w-5 h-5"
                style={{ color: "var(--text-secondary)" }}
              />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
