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
    if (score >= 85) return "text-green-400";
    if (score >= 70) return "text-yellow-400";
    return "text-red-400";
  };

  const getTrendIcon = (trend: "up" | "down" | "stable") => {
    switch (trend) {
      case "up":
        return <TrendingUp className="w-4 h-4 text-green-400" />;
      case "down":
        return <TrendingDown className="w-4 h-4 text-red-400" />;
      default:
        return <div className="w-4 h-4 bg-gray-400 rounded-full" />;
    }
  };

  return (
    <div className="bg-slate-900/50 rounded-xl p-5 hover:bg-slate-900/70 transition-all">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4 flex-1">
          <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
            <span className="text-white font-bold text-lg">
              {report.agentName.charAt(0)}
            </span>
          </div>
          <div className="flex-1">
            <div className="flex items-center space-x-3 mb-1">
              <p className="text-white font-semibold text-lg">
                {report.agentName}
              </p>
              <Badge
                variant={report.type === "weekly" ? "blue" : "purple"}
                size="sm"
              >
                {report.type.charAt(0).toUpperCase() + report.type.slice(1)}
              </Badge>
            </div>
            <div className="flex items-center space-x-4 text-sm text-gray-400">
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
            <p className="text-gray-400 text-xs mb-1">Score</p>
            <p
              className={`text-2xl font-bold ${getScoreColor(
                report.overallScore
              )}`}
            >
              {report.overallScore}%
            </p>
          </div>
          <div className="text-center">
            <p className="text-gray-400 text-xs mb-1">Calls</p>
            <p className="text-white font-semibold text-lg">
              {report.totalCalls}
            </p>
          </div>
          <div className="text-center">
            <p className="text-gray-400 text-xs mb-1">Avg Duration</p>
            <p className="text-white font-semibold">{report.avgCallDuration}</p>
          </div>
          <div className="flex flex-col items-center">
            <p className="text-gray-400 text-xs mb-1">Trend</p>
            {getTrendIcon(report.trend)}
          </div>
          <div className="flex space-x-2">
            <button
              onClick={onView}
              className="p-2 bg-purple-500/20 hover:bg-purple-500/30 rounded-lg transition-colors"
            >
              <Eye className="w-5 h-5 text-purple-400" />
            </button>
            <button
              onClick={onDownload}
              className="p-2 bg-blue-500/20 hover:bg-blue-500/30 rounded-lg transition-colors"
            >
              <Download className="w-5 h-5 text-blue-400" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
