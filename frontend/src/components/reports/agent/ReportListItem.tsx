import {
  TrendingUp,
  TrendingDown,
  Calendar,
  Award,
  Target,
  Download,
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
        return "bg-blue-500/20 text-blue-400";
      case "monthly":
        return "bg-purple-500/20 text-purple-400";
      case "quarterly":
        return "bg-pink-500/20 text-pink-400";
      default:
        return "bg-gray-500/20 text-gray-400";
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 85) return "text-green-400";
    if (score >= 70) return "text-yellow-400";
    return "text-red-400";
  };

  return (
    <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6 hover:border-blue-500/30 transition-all">
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <div className="flex items-center space-x-3 mb-2">
            <span
              className={`px-3 py-1 rounded-full text-xs font-semibold ${getTypeColor(
                report.type
              )}`}
            >
              {report.type.toUpperCase()}
            </span>
            <h3 className="text-xl font-bold text-white">{report.period}</h3>
          </div>
          <p className="text-gray-400 flex items-center">
            <Calendar className="w-4 h-4 mr-2" />
            {report.date}
          </p>
        </div>
        <div className="text-right">
          <div className="flex items-center justify-end space-x-2 mb-2">
            <span
              className={`text-3xl font-bold ${getScoreColor(report.score)}`}
            >
              {report.score}
            </span>
            {report.trend === "up" ? (
              <TrendingUp className="w-6 h-6 text-green-400" />
            ) : report.trend === "down" ? (
              <TrendingDown className="w-6 h-6 text-red-400" />
            ) : (
              <div className="w-6 h-1 bg-gray-400 rounded" />
            )}
          </div>
          <p className="text-gray-400 text-sm">Performance Score</p>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-slate-900/50 rounded-lg p-4">
          <p className="text-gray-400 text-sm mb-1">Total Calls</p>
          <p className="text-2xl font-bold text-white">{report.totalCalls}</p>
        </div>
        <div className="bg-slate-900/50 rounded-lg p-4">
          <p className="text-gray-400 text-sm mb-1">Avg Duration</p>
          <p className="text-2xl font-bold text-white">
            {Math.floor(report.avgDuration / 60)}:
            {String(report.avgDuration % 60).padStart(2, "0")}
          </p>
        </div>
      </div>

      {/* Strengths and Improvements */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div>
          <h4 className="text-white font-semibold mb-2 flex items-center">
            <Award className="w-4 h-4 mr-2 text-green-400" />
            Strengths
          </h4>
          <ul className="space-y-1">
            {report.strengths.map((strength, idx) => (
              <li key={idx} className="text-gray-400 text-sm flex items-center">
                <span className="w-1.5 h-1.5 bg-green-400 rounded-full mr-2" />
                {strength}
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h4 className="text-white font-semibold mb-2 flex items-center">
            <Target className="w-4 h-4 mr-2 text-yellow-400" />
            Areas for Improvement
          </h4>
          <ul className="space-y-1">
            {report.improvements.map((improvement, idx) => (
              <li key={idx} className="text-gray-400 text-sm flex items-center">
                <span className="w-1.5 h-1.5 bg-yellow-400 rounded-full mr-2" />
                {improvement}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center space-x-3 pt-4 border-t border-white/10">
        <button
          onClick={() => onViewDetails(report)}
          className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
        >
          <Eye className="w-4 h-4" />
          <span>View Details</span>
        </button>
        <button className="flex items-center space-x-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors">
          <Download className="w-4 h-4" />
          <span>Download PDF</span>
        </button>
      </div>
    </div>
  );
}
