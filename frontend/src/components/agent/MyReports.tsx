"use client";

import { useState } from "react";
import {
  FileText,
  TrendingUp,
  TrendingDown,
  Calendar,
  Award,
  Target,
  Filter,
  Search,
  Download,
  Eye,
} from "lucide-react";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

interface Report {
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
}

export default function MyReports() {
  const [filter, setFilter] = useState<"all" | "weekly" | "monthly">("all");
  const [searchQuery, setSearchQuery] = useState("");

  const [reports, setReports] = useState<Report[]>([
    {
      id: 1,
      type: "weekly",
      period: "Week 50, 2024",
      date: "Dec 10 - Dec 16",
      score: 87,
      trend: "up",
      totalCalls: 24,
      avgDuration: 456,
      strengths: ["Active listening", "Empathy", "Problem resolution"],
      improvements: ["Call opening", "Product knowledge"],
    },
    {
      id: 2,
      type: "weekly",
      period: "Week 49, 2024",
      date: "Dec 3 - Dec 9",
      score: 84,
      trend: "up",
      totalCalls: 22,
      avgDuration: 432,
      strengths: ["Customer engagement", "Clear communication"],
      improvements: ["Follow-up questions", "Time management"],
    },
    {
      id: 3,
      type: "monthly",
      period: "November 2024",
      date: "Nov 1 - Nov 30",
      score: 82,
      trend: "stable",
      totalCalls: 89,
      avgDuration: 428,
      strengths: ["Professionalism", "Patience"],
      improvements: ["Technical knowledge", "Objection handling"],
    },
    {
      id: 4,
      type: "weekly",
      period: "Week 48, 2024",
      date: "Nov 26 - Dec 2",
      score: 81,
      trend: "down",
      totalCalls: 19,
      avgDuration: 445,
      strengths: ["Tone management", "Closing techniques"],
      improvements: ["Active listening", "Clarifying questions"],
    },
  ]);

  const performanceData = [
    { month: "Jul", score: 75, calls: 78 },
    { month: "Aug", score: 78, calls: 82 },
    { month: "Sep", score: 80, calls: 85 },
    { month: "Oct", score: 82, calls: 89 },
    { month: "Nov", score: 82, calls: 89 },
    { month: "Dec", score: 87, calls: 67 },
  ];

  const categoryScores = [
    { category: "Empathy", score: 92 },
    { category: "Communication", score: 88 },
    { category: "Problem Solving", score: 85 },
    { category: "Product Knowledge", score: 79 },
    { category: "Time Management", score: 83 },
  ];

  const filteredReports = reports.filter((report) => {
    if (filter !== "all" && report.type !== filter) return false;
    if (
      searchQuery &&
      !report.period.toLowerCase().includes(searchQuery.toLowerCase())
    ) {
      return false;
    }
    return true;
  });

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

  const avgScore =
    reports.reduce((acc, r) => acc + r.score, 0) / reports.length;
  const totalCalls = reports.reduce((acc, r) => acc + r.totalCalls, 0);
  const latestScore = reports[0].score;
  const scoreTrend = latestScore - reports[1].score;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-pink-600 rounded-2xl p-8 text-white">
        <h1 className="text-3xl font-bold mb-2">My Performance Reports</h1>
        <p className="text-purple-100 mb-6">
          Track your progress and identify areas for improvement
        </p>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
            <p className="text-purple-100 text-sm mb-1">Current Score</p>
            <p className="text-3xl font-bold">{latestScore}</p>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
            <p className="text-purple-100 text-sm mb-1">Average Score</p>
            <p className="text-3xl font-bold">{Math.round(avgScore)}</p>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
            <p className="text-purple-100 text-sm mb-1">Total Reports</p>
            <p className="text-3xl font-bold">{reports.length}</p>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
            <p className="text-purple-100 text-sm mb-1">Trend</p>
            <div className="flex items-center space-x-2">
              <p className="text-3xl font-bold">
                {scoreTrend > 0 ? "+" : ""}
                {scoreTrend}
              </p>
              {scoreTrend > 0 ? (
                <TrendingUp className="w-6 h-6 text-green-300" />
              ) : (
                <TrendingDown className="w-6 h-6 text-red-300" />
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Performance Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Score Trend */}
        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <h3 className="text-xl font-bold text-white mb-4 flex items-center">
            <TrendingUp className="w-5 h-5 mr-2 text-blue-400" />
            Performance Trend
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={performanceData}>
              <defs>
                <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="month" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" domain={[0, 100]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1e293b",
                  border: "1px solid #334155",
                  borderRadius: "8px",
                }}
              />
              <Area
                type="monotone"
                dataKey="score"
                stroke="#3b82f6"
                fillOpacity={1}
                fill="url(#colorScore)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Category Scores */}
        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <h3 className="text-xl font-bold text-white mb-4 flex items-center">
            <Target className="w-5 h-5 mr-2 text-purple-400" />
            Skill Categories
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={categoryScores} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" domain={[0, 100]} stroke="#94a3b8" />
              <YAxis
                dataKey="category"
                type="category"
                stroke="#94a3b8"
                width={120}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1e293b",
                  border: "1px solid #334155",
                  borderRadius: "8px",
                }}
              />
              <Bar dataKey="score" fill="#8b5cf6" radius={[0, 8, 8, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Filters and Search */}
      <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div className="flex items-center space-x-2">
            <Filter className="w-5 h-5 text-gray-400" />
            <div className="flex space-x-2">
              {["all", "weekly", "monthly"].map((type) => (
                <button
                  key={type}
                  onClick={() => setFilter(type as any)}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    filter === type
                      ? "bg-blue-600 text-white"
                      : "bg-slate-700/50 text-gray-400 hover:bg-slate-700"
                  }`}
                >
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </button>
              ))}
            </div>
          </div>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search reports..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 pr-4 py-2 bg-slate-700/50 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>
      </div>

      {/* Reports List */}
      <div className="space-y-4">
        {filteredReports.map((report) => (
          <div
            key={report.id}
            className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6 hover:border-blue-500/30 transition-all"
          >
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
                  <h3 className="text-xl font-bold text-white">
                    {report.period}
                  </h3>
                </div>
                <p className="text-gray-400 flex items-center">
                  <Calendar className="w-4 h-4 mr-2" />
                  {report.date}
                </p>
              </div>
              <div className="text-right">
                <div className="flex items-center justify-end space-x-2 mb-2">
                  <span
                    className={`text-3xl font-bold ${getScoreColor(
                      report.score
                    )}`}
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
                <p className="text-2xl font-bold text-white">
                  {report.totalCalls}
                </p>
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
                    <li
                      key={idx}
                      className="text-gray-400 text-sm flex items-center"
                    >
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
                    <li
                      key={idx}
                      className="text-gray-400 text-sm flex items-center"
                    >
                      <span className="w-1.5 h-1.5 bg-yellow-400 rounded-full mr-2" />
                      {improvement}
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Actions */}
            <div className="flex items-center space-x-3 pt-4 border-t border-white/10">
              <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
                <Eye className="w-4 h-4" />
                <span>View Details</span>
              </button>
              <button className="flex items-center space-x-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors">
                <Download className="w-4 h-4" />
                <span>Download PDF</span>
              </button>
            </div>
          </div>
        ))}
      </div>

      {filteredReports.length === 0 && (
        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-12 text-center">
          <FileText className="w-16 h-16 text-gray-600 mx-auto mb-4" />
          <h3 className="text-xl font-bold text-white mb-2">
            No reports found
          </h3>
          <p className="text-gray-400">
            Try adjusting your filters or search query
          </p>
        </div>
      )}
    </div>
  );
}
