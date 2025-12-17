"use client";

import { useEffect, useState } from "react";
import {
  Phone,
  FileText,
  TrendingUp,
  Clock,
  Award,
  Target,
  Activity,
  Upload,
  CheckCircle,
} from "lucide-react";
import Link from "next/link";

interface DashboardStats {
  totalCalls: number;
  avgDuration: number;
  recentReports: number;
  performanceScore: number;
  todayCalls: number;
  weekCalls: number;
}

export default function AgentDashboard() {
  const [stats, setStats] = useState<DashboardStats>({
    totalCalls: 24,
    avgDuration: 456,
    recentReports: 3,
    performanceScore: 87,
    todayCalls: 5,
    weekCalls: 18,
  });

  const [recentCalls, setRecentCalls] = useState([
    { id: 1, date: "2 hours ago", duration: 456, score: 92 },
    { id: 2, date: "5 hours ago", duration: 723, score: 85 },
    { id: 3, date: "Yesterday", duration: 312, score: 78 },
  ]);

  const [latestReport, setLatestReport] = useState({
    type: "Weekly Report",
    date: "Dec 10 - Dec 16",
    overallRating: "Excellent",
    score: 87,
    strengths: [
      "Strong active listening skills",
      "Excellent response time",
      "Professional demeanor",
    ],
    improvements: [
      "Could use more open-ended questions",
      "Work on call closing techniques",
    ],
  });

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="space-y-6">
      {/* Welcome Section */}
      <div className="bg-gradient-to-r from-blue-600 to-cyan-600 rounded-2xl p-8 text-white">
        <h1 className="text-3xl font-bold mb-2">Welcome back, Agent!</h1>
        <p className="text-blue-100 mb-6">
          Here's your performance overview for this week
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
            <p className="text-blue-100 text-sm mb-1">Today's Calls</p>
            <p className="text-3xl font-bold">{stats.todayCalls}</p>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
            <p className="text-blue-100 text-sm mb-1">This Week</p>
            <p className="text-3xl font-bold">{stats.weekCalls}</p>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
            <p className="text-blue-100 text-sm mb-1">Performance</p>
            <p className="text-3xl font-bold">{stats.performanceScore}%</p>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Link
          href="/dashboard/upload-call"
          className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6 hover:bg-slate-800/70 transition-all group"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
              <Upload className="w-6 h-6 text-blue-400" />
            </div>
            <span className="text-blue-400 text-sm font-medium">Upload</span>
          </div>
          <h3 className="text-lg font-semibold text-white mb-2">
            Upload New Call
          </h3>
          <p className="text-gray-400 text-sm">
            Upload and analyze a new call recording
          </p>
        </Link>

        <Link
          href="/dashboard/my-reports"
          className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6 hover:bg-slate-800/70 transition-all group"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-cyan-500/20 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
              <FileText className="w-6 h-6 text-cyan-400" />
            </div>
            <span className="text-cyan-400 text-sm font-medium">View</span>
          </div>
          <h3 className="text-lg font-semibold text-white mb-2">My Reports</h3>
          <p className="text-gray-400 text-sm">
            View and analyze all your call recordings
          </p>
        </Link>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
              <Phone className="w-5 h-5 text-blue-400" />
            </div>
          </div>
          <p className="text-gray-400 text-sm mb-1">Total Calls</p>
          <p className="text-3xl font-bold text-white">{stats.totalCalls}</p>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="w-10 h-10 bg-cyan-500/20 rounded-lg flex items-center justify-center">
              <Clock className="w-5 h-5 text-cyan-400" />
            </div>
          </div>
          <p className="text-gray-400 text-sm mb-1">Avg Duration</p>
          <p className="text-3xl font-bold text-white">
            {formatDuration(stats.avgDuration)}
          </p>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center">
              <FileText className="w-5 h-5 text-purple-400" />
            </div>
          </div>
          <p className="text-gray-400 text-sm mb-1">Reports</p>
          <p className="text-3xl font-bold text-white">{stats.recentReports}</p>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
              <Award className="w-5 h-5 text-green-400" />
            </div>
          </div>
          <p className="text-gray-400 text-sm mb-1">Performance</p>
          <p className="text-3xl font-bold text-white">
            {stats.performanceScore}%
          </p>
        </div>
      </div>

      {/* Recent Activity and Latest Report */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Calls */}
        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-white flex items-center">
              <Activity className="w-5 h-5 mr-2 text-blue-400" />
              Recent Calls
            </h2>
            <Link
              href="/dashboard/calls"
              className="text-sm text-blue-400 hover:text-blue-300"
            >
              View All
            </Link>
          </div>
          <div className="space-y-4">
            {recentCalls.map((call) => (
              <div
                key={call.id}
                className="flex items-center justify-between p-4 bg-slate-900/50 rounded-lg hover:bg-slate-900/70 transition-colors"
              >
                <div className="flex items-center space-x-4">
                  <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full flex items-center justify-center">
                    <Phone className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <p className="text-white font-medium">Call #{call.id}</p>
                    <p className="text-gray-400 text-sm">{call.date}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-white font-medium">
                    {formatDuration(call.duration)}
                  </p>
                  <p className="text-green-400 text-sm">{call.score}% score</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Latest Performance Report */}
        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-white flex items-center">
              <Target className="w-5 h-5 mr-2 text-purple-400" />
              Latest Report
            </h2>
            <Link
              href="/dashboard/reports"
              className="text-sm text-blue-400 hover:text-blue-300"
            >
              View All
            </Link>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-gradient-to-r from-purple-500/20 to-pink-500/20 border border-purple-500/30 rounded-lg">
              <div>
                <p className="text-white font-medium">{latestReport.type}</p>
                <p className="text-gray-300 text-sm">{latestReport.date}</p>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold text-white">
                  {latestReport.score}%
                </p>
                <p className="text-green-400 text-sm">
                  {latestReport.overallRating}
                </p>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-gray-400 mb-3">
                Strengths
              </h3>
              <div className="space-y-2">
                {latestReport.strengths.map((strength, index) => (
                  <div key={index} className="flex items-start space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                    <p className="text-gray-300 text-sm">{strength}</p>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-gray-400 mb-3">
                Areas to Improve
              </h3>
              <div className="space-y-2">
                {latestReport.improvements.map((improvement, index) => (
                  <div key={index} className="flex items-start space-x-2">
                    <TrendingUp className="w-4 h-4 text-orange-400 mt-0.5 flex-shrink-0" />
                    <p className="text-gray-300 text-sm">{improvement}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
