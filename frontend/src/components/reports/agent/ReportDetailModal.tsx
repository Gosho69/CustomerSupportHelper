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
    return "";
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
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/20" onClick={onClose} />
      <div
        className="relative rounded-lg p-8 w-full max-w-4xl max-h-[90vh] overflow-y-auto"
        style={{ background: "#ffffff", border: "1px solid var(--border)" }}
      >
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-2 rounded-lg transition-colors"
          style={{ background: "var(--accent-bg)" }}
        >
          <span className="text-2xl" style={{ color: "var(--text-secondary)" }}>
            ×
          </span>
        </button>

        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center space-x-3 mb-4">
            <span
              className={`px-3 py-1 rounded-full text-sm font-semibold ${getTypeColor(
                report.type,
              )}`}
              style={getTypeStyle(report.type)}
            >
              {report.type.toUpperCase()} REPORT
            </span>
          </div>
          <h2
            className="text-3xl font-bold mb-2"
            style={{ color: "var(--text-primary)" }}
          >
            {report.period}
          </h2>
          <p
            className="flex items-center"
            style={{ color: "var(--text-secondary)" }}
          >
            <Calendar className="w-4 h-4 mr-2" />
            {report.date}
          </p>
        </div>

        {/* Score Section */}
        <div
          className="rounded-lg p-6 mb-8"
          style={{
            background: "var(--background)",
            border: "1px solid var(--border)",
          }}
        >
          <div className="text-center">
            <p
              className="text-sm mb-2"
              style={{ color: "var(--text-secondary)" }}
            >
              Overall Performance Score
            </p>
            <div className="flex items-center justify-center space-x-3">
              <span
                className={`text-6xl font-bold ${getScoreColor(report.score)}`}
                style={getScoreStyle(report.score)}
              >
                {report.score}
              </span>
              {report.trend === "up" ? (
                <TrendingUp
                  className="w-8 h-8"
                  style={{ color: "var(--success, #0caf60)" }}
                />
              ) : report.trend === "down" ? (
                <TrendingDown
                  className="w-8 h-8"
                  style={{ color: "#e25950" }}
                />
              ) : (
                <div
                  className="w-8 h-2 rounded"
                  style={{ background: "var(--text-secondary)" }}
                />
              )}
            </div>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-4 mb-8">
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
              className="text-3xl font-bold"
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
              className="text-3xl font-bold"
              style={{ color: "var(--text-primary)" }}
            >
              {Math.floor(report.avgDuration / 60)}:
              {String(report.avgDuration % 60).padStart(2, "0")}
            </p>
          </div>
        </div>

        {/* Skills Breakdown */}
        {report.topSkills && (
          <div
            className="rounded-lg p-6 mb-8"
            style={{
              background: "var(--background)",
              border: "1px solid var(--border)",
            }}
          >
            <h3
              className="text-lg font-semibold mb-4 flex items-center"
              style={{ color: "var(--text-primary)" }}
            >
              <Target
                className="w-5 h-5 mr-2"
                style={{ color: "var(--accent)" }}
              />
              Skills Breakdown
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={report.topSkills}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ee" />
                <XAxis dataKey="skill" stroke="#697386" />
                <YAxis stroke="#697386" domain={[0, 100]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#ffffff",
                    border: "1px solid #e3e8ee",
                    borderRadius: "8px",
                    color: "#1a1f36",
                  }}
                />
                <Bar dataKey="score" fill="#635bff" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

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
                    className="mr-2 mt-0.5"
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
              <Target
                className="w-5 h-5 mr-2"
                style={{ color: "var(--warning, #e68a00)" }}
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
                  <span
                    style={{ color: "var(--warning, #e68a00)" }}
                    className="mr-2 mt-0.5"
                  >
                    →
                  </span>
                  {improvement}
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-4">
          <button
            className="flex-1 px-6 py-3 text-white rounded-lg font-semibold transition-all flex items-center justify-center space-x-2"
            style={{ background: "var(--accent)" }}
          >
            <Download className="w-5 h-5" />
            <span>Download PDF</span>
          </button>
          <button
            onClick={onClose}
            className="flex-1 px-6 py-3 rounded-lg font-semibold transition-all"
            style={{
              background: "var(--accent-bg)",
              color: "var(--text-primary)",
            }}
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
