import {
  Phone,
  Mail,
  Calendar,
  Award,
  TrendingUp,
  TrendingDown,
  Download,
  User,
} from "lucide-react";
import { Modal, Button, Badge, StatusIndicator } from "@/components/ui";
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
    <Modal isOpen={!!report} onClose={onClose} size="5xl">
      {/* Header */}
      <div className="flex items-start space-x-6 mb-8">
        <div className="w-20 h-20 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
          <span className="text-white font-bold text-3xl">
            {report.agentName.charAt(0)}
          </span>
        </div>
        <div className="flex-1">
          <div className="flex items-center space-x-3 mb-2">
            <h2 className="text-2xl font-bold text-white">
              {report.agentName}
            </h2>
            <Badge variant={report.type === "weekly" ? "blue" : "purple"}>
              {report.type.charAt(0).toUpperCase() + report.type.slice(1)}{" "}
              Report
            </Badge>
          </div>
          <p className="text-gray-400">{report.agentEmail}</p>
          <p className="text-gray-400 text-sm mt-1">Period: {report.period}</p>
        </div>
        <div className="text-right">
          <p className="text-gray-400 text-sm mb-1">Overall Score</p>
          <p
            className={`text-4xl font-bold ${getScoreColor(
              report.overallScore
            )}`}
          >
            {report.overallScore}%
          </p>
          {getTrendIcon(report.trend)}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 mb-8">
        <div className="bg-slate-900/50 rounded-xl p-4">
          <p className="text-gray-400 text-sm mb-1">Total Calls</p>
          <p className="text-2xl font-bold text-white">{report.totalCalls}</p>
        </div>
        <div className="bg-slate-900/50 rounded-xl p-4">
          <p className="text-gray-400 text-sm mb-1">Avg Duration</p>
          <p className="text-2xl font-bold text-white">
            {report.avgCallDuration}
          </p>
        </div>
        <div className="bg-slate-900/50 rounded-xl p-4">
          <p className="text-gray-400 text-sm mb-1">Trend</p>
          <p className="text-xl font-bold text-white capitalize">
            {report.trend}
          </p>
        </div>
      </div>

      {/* Skills Chart */}
      <div className="bg-slate-900/50 rounded-xl p-6 mb-8">
        <h3 className="text-lg font-semibold text-white mb-4">
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
        <div className="bg-slate-900/50 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <Award className="w-5 h-5 mr-2 text-green-400" />
            Strengths
          </h3>
          <ul className="space-y-2">
            {report.strengths.map((strength, index) => (
              <li
                key={index}
                className="flex items-start text-gray-300 text-sm"
              >
                <span className="text-green-400 mr-2">✓</span>
                {strength}
              </li>
            ))}
          </ul>
        </div>

        <div className="bg-slate-900/50 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <TrendingUp className="w-5 h-5 mr-2 text-yellow-400" />
            Areas for Improvement
          </h3>
          <ul className="space-y-2">
            {report.improvements.map((improvement, index) => (
              <li
                key={index}
                className="flex items-start text-gray-300 text-sm"
              >
                <span className="text-yellow-400 mr-2">→</span>
                {improvement}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-4">
        <Button icon={Download} iconPosition="left" className="flex-1">
          Download PDF
        </Button>
        <Button
          variant="secondary"
          icon={User}
          iconPosition="left"
          className="flex-1"
        >
          View Agent Profile
        </Button>
      </div>
    </Modal>
  );
}
