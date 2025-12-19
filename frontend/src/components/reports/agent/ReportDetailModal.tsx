"use client";

import {
  TrendingUp,
  TrendingDown,
  Calendar,
  Award,
  Target,
  Download,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { Report } from "./ReportListItem";

interface ReportDetailModalProps {
  report: Report;
  onClose: () => void;
}

export default function ReportDetailModal({
  report,
  onClose,
}: ReportDetailModalProps) {
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
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />
      <div className="relative bg-slate-800 rounded-2xl p-8 w-full max-w-4xl max-h-[90vh] overflow-y-auto border border-white/10">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-2 hover:bg-white/10 rounded-lg transition-colors"
        >
          <span className="text-gray-400 text-2xl">×</span>
        </button>

        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center space-x-3 mb-4">
            <span
              className={`px-3 py-1 rounded-full text-sm font-semibold ${getTypeColor(
                report.type
              )}`}
            >
              {report.type.toUpperCase()} REPORT
            </span>
          </div>
          <h2 className="text-3xl font-bold text-white mb-2">
            {report.period}
          </h2>
          <p className="text-gray-400 flex items-center">
            <Calendar className="w-4 h-4 mr-2" />
            {report.date}
          </p>
        </div>

        {/* Score Section */}
        <div className="bg-slate-900/50 rounded-xl p-6 mb-8">
          <div className="text-center">
            <p className="text-gray-400 text-sm mb-2">
              Overall Performance Score
            </p>
            <div className="flex items-center justify-center space-x-3">
              <span
                className={`text-6xl font-bold ${getScoreColor(report.score)}`}
              >
                {report.score}
              </span>
              {report.trend === "up" ? (
                <TrendingUp className="w-8 h-8 text-green-400" />
              ) : report.trend === "down" ? (
                <TrendingDown className="w-8 h-8 text-red-400" />
              ) : (
                <div className="w-8 h-2 bg-gray-400 rounded" />
              )}
            </div>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-4 mb-8">
          <div className="bg-slate-900/50 rounded-xl p-4">
            <p className="text-gray-400 text-sm mb-1">Total Calls</p>
            <p className="text-3xl font-bold text-white">{report.totalCalls}</p>
          </div>
          <div className="bg-slate-900/50 rounded-xl p-4">
            <p className="text-gray-400 text-sm mb-1">Avg Duration</p>
            <p className="text-3xl font-bold text-white">
              {Math.floor(report.avgDuration / 60)}:
              {String(report.avgDuration % 60).padStart(2, "0")}
            </p>
          </div>
        </div>

        {/* Skills Breakdown */}
        {report.topSkills && (
          <div className="bg-slate-900/50 rounded-xl p-6 mb-8">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <Target className="w-5 h-5 mr-2 text-purple-400" />
              Skills Breakdown
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={report.topSkills}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                <XAxis dataKey="skill" stroke="#9ca3af" />
                <YAxis stroke="#9ca3af" domain={[0, 100]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#1e293b",
                    border: "1px solid #ffffff20",
                    borderRadius: "8px",
                  }}
                />
                <Bar dataKey="score" fill="#a855f7" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

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
                  <span className="text-green-400 mr-2 mt-0.5">✓</span>
                  {strength}
                </li>
              ))}
            </ul>
          </div>

          <div className="bg-slate-900/50 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <Target className="w-5 h-5 mr-2 text-yellow-400" />
              Areas for Improvement
            </h3>
            <ul className="space-y-2">
              {report.improvements.map((improvement, index) => (
                <li
                  key={index}
                  className="flex items-start text-gray-300 text-sm"
                >
                  <span className="text-yellow-400 mr-2 mt-0.5">→</span>
                  {improvement}
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-4">
          <button className="flex-1 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white rounded-xl font-semibold transition-all flex items-center justify-center space-x-2">
            <Download className="w-5 h-5" />
            <span>Download PDF</span>
          </button>
          <button
            onClick={onClose}
            className="flex-1 px-6 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-xl font-semibold transition-all"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
