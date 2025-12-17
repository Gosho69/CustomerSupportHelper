"use client";

import { useEffect, useState } from "react";
import {
  Users,
  FileText,
  TrendingUp,
  Phone,
  Award,
  AlertCircle,
  Target,
  Activity,
  UserPlus,
  BarChart3,
} from "lucide-react";
import Link from "next/link";

interface TeamMember {
  id: number;
  name: string;
  email: string;
  totalCalls: number;
  avgScore: number;
  trend: "up" | "down" | "stable";
}

interface DashboardStats {
  totalTeamMembers: number;
  totalTeamCalls: number;
  avgTeamPerformance: number;
  reportsGenerated: number;
  todayTeamCalls: number;
  weekTeamCalls: number;
}

export default function HeadOfDepartmentDashboard() {
  const [stats, setStats] = useState<DashboardStats>({
    totalTeamMembers: 8,
    totalTeamCalls: 156,
    avgTeamPerformance: 84,
    reportsGenerated: 12,
    todayTeamCalls: 23,
    weekTeamCalls: 98,
  });

  const [teamMembers, setTeamMembers] = useState<TeamMember[]>([
    {
      id: 1,
      name: "John Smith",
      email: "john.smith@example.com",
      totalCalls: 45,
      avgScore: 92,
      trend: "up",
    },
    {
      id: 2,
      name: "Sarah Johnson",
      email: "sarah.j@example.com",
      totalCalls: 38,
      avgScore: 88,
      trend: "up",
    },
    {
      id: 3,
      name: "Mike Wilson",
      email: "mike.w@example.com",
      totalCalls: 42,
      avgScore: 76,
      trend: "down",
    },
    {
      id: 4,
      name: "Emily Davis",
      email: "emily.d@example.com",
      totalCalls: 31,
      avgScore: 85,
      trend: "stable",
    },
  ]);

  const [recentReports, setRecentReports] = useState([
    {
      id: 1,
      agentName: "John Smith",
      type: "Weekly",
      date: "Dec 10 - Dec 16",
      score: 92,
    },
    {
      id: 2,
      agentName: "Sarah Johnson",
      type: "Weekly",
      date: "Dec 10 - Dec 16",
      score: 88,
    },
    {
      id: 3,
      agentName: "Mike Wilson",
      type: "Monthly",
      date: "November 2024",
      score: 76,
    },
  ]);

  const [alerts, setAlerts] = useState([
    {
      id: 1,
      type: "warning",
      message: "Mike Wilson's performance dropped by 12% this week",
      time: "2 hours ago",
    },
    {
      id: 2,
      type: "info",
      message: "3 new reports ready for review",
      time: "5 hours ago",
    },
  ]);

  const getTrendIcon = (trend: "up" | "down" | "stable") => {
    switch (trend) {
      case "up":
        return <TrendingUp className="w-4 h-4 text-green-400" />;
      case "down":
        return <TrendingUp className="w-4 h-4 text-red-400 rotate-180" />;
      default:
        return <div className="w-4 h-4 bg-gray-400 rounded-full" />;
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 85) return "text-green-400";
    if (score >= 70) return "text-yellow-400";
    return "text-red-400";
  };

  return (
    <div className="space-y-6">
      {/* Welcome Section */}
      <div className="bg-gradient-to-r from-purple-600 to-pink-600 rounded-2xl p-8 text-white">
        <h1 className="text-3xl font-bold mb-2">Team Performance Overview</h1>
        <p className="text-purple-100 mb-6">
          Monitor and manage your team's performance
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
            <p className="text-purple-100 text-sm mb-1">Team Members</p>
            <p className="text-3xl font-bold">{stats.totalTeamMembers}</p>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
            <p className="text-purple-100 text-sm mb-1">This Week's Calls</p>
            <p className="text-3xl font-bold">{stats.weekTeamCalls}</p>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
            <p className="text-purple-100 text-sm mb-1">Avg Performance</p>
            <p className="text-3xl font-bold">{stats.avgTeamPerformance}%</p>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Link
          href="/dashboard/team"
          className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6 hover:bg-slate-800/70 transition-all group"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
              <Users className="w-6 h-6 text-blue-400" />
            </div>
            <span className="text-blue-400 text-sm font-medium">Manage</span>
          </div>
          <h3 className="text-lg font-semibold text-white mb-2">My Team</h3>
          <p className="text-gray-400 text-sm">View and manage team members</p>
        </Link>

        <Link
          href="/dashboard/calls"
          className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6 hover:bg-slate-800/70 transition-all group"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-cyan-500/20 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
              <Phone className="w-6 h-6 text-cyan-400" />
            </div>
            <span className="text-cyan-400 text-sm font-medium">View All</span>
          </div>
          <h3 className="text-lg font-semibold text-white mb-2">Team Calls</h3>
          <p className="text-gray-400 text-sm">
            Review all team call recordings
          </p>
        </Link>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
              <Users className="w-5 h-5 text-blue-400" />
            </div>
          </div>
          <p className="text-gray-400 text-sm mb-1">Team Members</p>
          <p className="text-3xl font-bold text-white">
            {stats.totalTeamMembers}
          </p>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="w-10 h-10 bg-cyan-500/20 rounded-lg flex items-center justify-center">
              <Phone className="w-5 h-5 text-cyan-400" />
            </div>
          </div>
          <p className="text-gray-400 text-sm mb-1">Total Calls</p>
          <p className="text-3xl font-bold text-white">
            {stats.totalTeamCalls}
          </p>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center">
              <FileText className="w-5 h-5 text-purple-400" />
            </div>
          </div>
          <p className="text-gray-400 text-sm mb-1">Reports</p>
          <p className="text-3xl font-bold text-white">
            {stats.reportsGenerated}
          </p>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
              <Award className="w-5 h-5 text-green-400" />
            </div>
          </div>
          <p className="text-gray-400 text-sm mb-1">Avg Performance</p>
          <p className="text-3xl font-bold text-white">
            {stats.avgTeamPerformance}%
          </p>
        </div>
      </div>

      {/* Team Performance and Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Team Members Performance */}
        <div className="lg:col-span-2 bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-white flex items-center">
              <BarChart3 className="w-5 h-5 mr-2 text-blue-400" />
              Team Performance
            </h2>
            <Link
              href="/dashboard/team"
              className="text-sm text-blue-400 hover:text-blue-300"
            >
              View All
            </Link>
          </div>
          <div className="space-y-4">
            {teamMembers.map((member) => (
              <div
                key={member.id}
                className="flex items-center justify-between p-4 bg-slate-900/50 rounded-lg hover:bg-slate-900/70 transition-colors"
              >
                <div className="flex items-center space-x-4 flex-1">
                  <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
                    <span className="text-white font-medium text-sm">
                      {member.name.charAt(0)}
                    </span>
                  </div>
                  <div className="flex-1">
                    <p className="text-white font-medium">{member.name}</p>
                    <p className="text-gray-400 text-sm">{member.email}</p>
                  </div>
                </div>
                <div className="flex items-center space-x-6">
                  <div className="text-right">
                    <p className="text-white font-medium">
                      {member.totalCalls} calls
                    </p>
                    <p
                      className={`text-sm font-medium ${getScoreColor(
                        member.avgScore
                      )}`}
                    >
                      {member.avgScore}% avg
                    </p>
                  </div>
                  <div className="flex items-center space-x-2">
                    {getTrendIcon(member.trend)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Alerts & Notifications */}
        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <h2 className="text-xl font-bold text-white mb-6 flex items-center">
            <AlertCircle className="w-5 h-5 mr-2 text-orange-400" />
            Alerts
          </h2>
          <div className="space-y-4">
            {alerts.map((alert) => (
              <div
                key={alert.id}
                className={`p-4 rounded-lg border ${
                  alert.type === "warning"
                    ? "bg-orange-500/10 border-orange-500/30"
                    : "bg-blue-500/10 border-blue-500/30"
                }`}
              >
                <div className="flex items-start space-x-3">
                  <AlertCircle
                    className={`w-5 h-5 mt-0.5 ${
                      alert.type === "warning"
                        ? "text-orange-400"
                        : "text-blue-400"
                    }`}
                  />
                  <div className="flex-1">
                    <p className="text-white text-sm mb-1">{alert.message}</p>
                    <p className="text-gray-400 text-xs">{alert.time}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-6">
            <Link
              href="/dashboard/team"
              className="w-full px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white text-sm font-medium rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all duration-200 shadow-lg shadow-purple-500/30 flex items-center justify-center space-x-2"
            >
              <UserPlus className="w-4 h-4" />
              <span>Add Team Member</span>
            </Link>
          </div>
        </div>
      </div>

      {/* Recent Reports */}
      <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-white flex items-center">
            <Target className="w-5 h-5 mr-2 text-purple-400" />
            Recent Reports
          </h2>
          <Link
            href="/dashboard/reports"
            className="text-sm text-blue-400 hover:text-blue-300"
          >
            View All
          </Link>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {recentReports.map((report) => (
            <div
              key={report.id}
              className="p-4 bg-slate-900/50 rounded-lg hover:bg-slate-900/70 transition-colors"
            >
              <div className="flex items-center justify-between mb-3">
                <span className="px-3 py-1 bg-purple-500/20 text-purple-400 text-xs font-medium rounded-full">
                  {report.type}
                </span>
                <span
                  className={`text-xl font-bold ${getScoreColor(report.score)}`}
                >
                  {report.score}%
                </span>
              </div>
              <p className="text-white font-medium mb-1">{report.agentName}</p>
              <p className="text-gray-400 text-sm">{report.date}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
