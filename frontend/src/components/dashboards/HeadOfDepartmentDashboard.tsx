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
import { PageHeader } from "@/components/ui";

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
        return (
          <TrendingUp className="w-4 h-4" style={{ color: "var(--success)" }} />
        );
      case "down":
        return (
          <TrendingUp
            className="w-4 h-4 rotate-180"
            style={{ color: "var(--danger)" }}
          />
        );
      default:
        return (
          <div
            className="w-4 h-4 rounded-full"
            style={{ background: "var(--text-tertiary)" }}
          />
        );
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 85) return "var(--success)";
    if (score >= 70) return "var(--warning)";
    return "var(--danger)";
  };

  const cardStyle: React.CSSProperties = {
    background: "#ffffff",
    border: "1px solid var(--border, #e3e8ee)",
    borderRadius: "8px",
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Team Performance Overview"
        subtitle="Monitor and manage your team's performance"
      />

      {/* Top Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="rounded-lg p-6" style={cardStyle}>
          <p
            className="mb-1 text-sm font-medium"
            style={{ color: "var(--text-secondary)" }}
          >
            Team Members
          </p>
          <p
            className="text-3xl font-bold"
            style={{ color: "var(--text-primary)" }}
          >
            {stats.totalTeamMembers}
          </p>
        </div>
        <div className="rounded-lg p-6" style={cardStyle}>
          <p
            className="mb-1 text-sm font-medium"
            style={{ color: "var(--text-secondary)" }}
          >
            This Week's Calls
          </p>
          <p
            className="text-3xl font-bold"
            style={{ color: "var(--text-primary)" }}
          >
            {stats.weekTeamCalls}
          </p>
        </div>
        <div className="rounded-lg p-6" style={cardStyle}>
          <p
            className="mb-1 text-sm font-medium"
            style={{ color: "var(--text-secondary)" }}
          >
            Avg Performance
          </p>
          <p
            className="text-3xl font-bold"
            style={{ color: "var(--text-primary)" }}
          >
            {stats.avgTeamPerformance}%
          </p>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Link
          href="/dashboard/team"
          className="rounded-lg p-6 hover:shadow-md transition-shadow"
          style={cardStyle}
        >
          <div className="flex items-center">
            <div
              className="w-12 h-12 rounded-lg flex items-center justify-center"
              style={{ background: "var(--accent-bg)" }}
            >
              <Users className="w-6 h-6" style={{ color: "var(--accent)" }} />
            </div>
            <div className="ml-4">
              <p
                className="text-sm font-medium"
                style={{ color: "var(--text-primary)" }}
              >
                My Team
              </p>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                View and manage team members
              </p>
            </div>
          </div>
        </Link>

        <Link
          href="/dashboard/calls"
          className="rounded-lg p-6 hover:shadow-md transition-shadow"
          style={cardStyle}
        >
          <div className="flex items-center">
            <div
              className="w-12 h-12 rounded-lg flex items-center justify-center"
              style={{ background: "var(--accent-bg)" }}
            >
              <Phone className="w-6 h-6" style={{ color: "var(--accent)" }} />
            </div>
            <div className="ml-4">
              <p
                className="text-sm font-medium"
                style={{ color: "var(--text-primary)" }}
              >
                Team Calls
              </p>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                Review all team call recordings
              </p>
            </div>
          </div>
        </Link>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {[
          {
            icon: FileText,
            label: "Team Members",
            value: stats.totalTeamMembers,
          },
          { icon: Phone, label: "Total Calls", value: stats.totalTeamCalls },
          { icon: FileText, label: "Reports", value: stats.reportsGenerated },
          {
            icon: Award,
            label: "Avg Performance",
            value: `${stats.avgTeamPerformance}%`,
          },
        ].map((stat, index) => (
          <div key={index} className="rounded-lg p-6" style={cardStyle}>
            <div className="flex items-center justify-between mb-4">
              <div
                className="w-10 h-10 rounded-lg flex items-center justify-center"
                style={{ background: "var(--accent-bg)" }}
              >
                <stat.icon
                  className="w-5 h-5"
                  style={{ color: "var(--accent)" }}
                />
              </div>
            </div>
            <p
              className="text-sm mb-1"
              style={{ color: "var(--text-secondary)" }}
            >
              {stat.label}
            </p>
            <p
              className="text-3xl font-bold"
              style={{ color: "var(--text-primary)" }}
            >
              {stat.value}
            </p>
          </div>
        ))}
      </div>

      {/* Team Performance and Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Team Members Performance */}
        <div className="lg:col-span-2 rounded-lg p-6" style={cardStyle}>
          <div className="flex items-center justify-between mb-6">
            <h2
              className="text-lg font-semibold flex items-center"
              style={{ color: "var(--text-primary)" }}
            >
              <BarChart3
                className="w-5 h-5 mr-2"
                style={{ color: "var(--accent)" }}
              />
              Team Performance
            </h2>
            <Link
              href="/dashboard/team"
              className="text-sm font-medium"
              style={{ color: "var(--accent)" }}
            >
              View All
            </Link>
          </div>
          <div className="space-y-3">
            {teamMembers.map((member) => (
              <div
                key={member.id}
                className="flex items-center justify-between p-4 rounded-lg transition-colors"
                style={{ background: "var(--background)" }}
              >
                <div className="flex items-center space-x-4 flex-1">
                  <div
                    className="w-10 h-10 rounded-full flex items-center justify-center"
                    style={{ background: "var(--accent-bg)" }}
                  >
                    <span
                      className="font-medium text-sm"
                      style={{ color: "var(--accent)" }}
                    >
                      {member.name.charAt(0)}
                    </span>
                  </div>
                  <div className="flex-1">
                    <p
                      className="font-medium"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {member.name}
                    </p>
                    <p
                      className="text-sm"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {member.email}
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-6">
                  <div className="text-right">
                    <p
                      className="font-medium"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {member.totalCalls} calls
                    </p>
                    <p
                      className="text-sm font-medium"
                      style={{ color: getScoreColor(member.avgScore) }}
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
        <div className="rounded-lg p-6" style={cardStyle}>
          <h2
            className="text-lg font-semibold mb-6 flex items-center"
            style={{ color: "var(--text-primary)" }}
          >
            <AlertCircle
              className="w-5 h-5 mr-2"
              style={{ color: "var(--warning)" }}
            />
            Alerts
          </h2>
          <div className="space-y-4">
            {alerts.map((alert) => (
              <div
                key={alert.id}
                className="p-4 rounded-lg"
                style={{
                  background:
                    alert.type === "warning"
                      ? "var(--warning-bg, #fff8e6)"
                      : "var(--accent-bg)",
                  border: `1px solid ${alert.type === "warning" ? "var(--warning)" : "var(--accent)"}20`,
                }}
              >
                <div className="flex items-start space-x-3">
                  <AlertCircle
                    className="w-5 h-5 mt-0.5"
                    style={{
                      color:
                        alert.type === "warning"
                          ? "var(--warning)"
                          : "var(--accent)",
                    }}
                  />
                  <div className="flex-1">
                    <p
                      className="text-sm mb-1"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {alert.message}
                    </p>
                    <p
                      className="text-xs"
                      style={{ color: "var(--text-tertiary)" }}
                    >
                      {alert.time}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-6">
            <Link
              href="/dashboard/team"
              className="w-full px-4 py-2 text-sm font-medium rounded-lg flex items-center justify-center space-x-2 text-white"
              style={{ background: "var(--accent)" }}
            >
              <UserPlus className="w-4 h-4" />
              <span>Add Team Member</span>
            </Link>
          </div>
        </div>
      </div>

      {/* Recent Reports */}
      <div className="rounded-lg p-6" style={cardStyle}>
        <div className="flex items-center justify-between mb-6">
          <h2
            className="text-lg font-semibold flex items-center"
            style={{ color: "var(--text-primary)" }}
          >
            <Target
              className="w-5 h-5 mr-2"
              style={{ color: "var(--accent)" }}
            />
            Recent Reports
          </h2>
          <Link
            href="/dashboard/reports"
            className="text-sm font-medium"
            style={{ color: "var(--accent)" }}
          >
            View All
          </Link>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {recentReports.map((report) => (
            <div
              key={report.id}
              className="p-4 rounded-lg transition-colors"
              style={{ background: "var(--background)" }}
            >
              <div className="flex items-center justify-between mb-3">
                <span
                  className="px-3 py-1 text-xs font-medium rounded-full"
                  style={{
                    background: "var(--accent-bg)",
                    color: "var(--accent)",
                  }}
                >
                  {report.type}
                </span>
                <span
                  className="text-xl font-bold"
                  style={{ color: getScoreColor(report.score) }}
                >
                  {report.score}%
                </span>
              </div>
              <p
                className="font-medium mb-1"
                style={{ color: "var(--text-primary)" }}
              >
                {report.agentName}
              </p>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                {report.date}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
