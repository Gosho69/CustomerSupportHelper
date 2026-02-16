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
import { PageHeader } from "@/components/ui";

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

  const cardStyle: React.CSSProperties = {
    background: "#ffffff",
    border: "1px solid var(--border, #e3e8ee)",
    borderRadius: "8px",
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <PageHeader
          title={`Welcome back, Agent!`}
          subtitle={`Here's your performance overview for this week`}
        />
      </div>

      {/* Top Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="rounded-lg p-6" style={cardStyle}>
          <p
            className="mb-1 text-sm font-medium"
            style={{ color: "var(--text-secondary, #697386)" }}
          >
            Today's Calls
          </p>
          <p
            className="text-3xl font-bold"
            style={{ color: "var(--text-primary, #1a1f36)" }}
          >
            {stats.todayCalls}
          </p>
        </div>
        <div className="rounded-lg p-6" style={cardStyle}>
          <p
            className="mb-1 text-sm font-medium"
            style={{ color: "var(--text-secondary, #697386)" }}
          >
            This Week
          </p>
          <p
            className="text-3xl font-bold"
            style={{ color: "var(--text-primary, #1a1f36)" }}
          >
            {stats.weekCalls}
          </p>
        </div>
        <div className="rounded-lg p-6" style={cardStyle}>
          <p
            className="mb-1 text-sm font-medium"
            style={{ color: "var(--text-secondary, #697386)" }}
          >
            Performance
          </p>
          <p
            className="text-3xl font-bold"
            style={{ color: "var(--accent, #635bff)" }}
          >
            {stats.performanceScore}%
          </p>
        </div>
      </div>

      {/* Main Layout: left column for quick actions & stats, right column for activity */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          {/* Quick Actions */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Link
              href="/dashboard/upload-call"
              className="rounded-lg p-6 transition-shadow hover:shadow-md"
              style={cardStyle}
            >
              <div className="flex items-center mb-4">
                <div
                  className="w-12 h-12 rounded-lg flex items-center justify-center"
                  style={{
                    background: "var(--accent-bg, #f0efff)",
                  }}
                >
                  <Upload
                    className="w-6 h-6"
                    style={{ color: "var(--accent, #635bff)" }}
                  />
                </div>
                <div className="ml-4">
                  <p
                    className="text-sm font-semibold"
                    style={{ color: "var(--text-primary, #1a1f36)" }}
                  >
                    Upload New Call
                  </p>
                  <p
                    className="text-sm"
                    style={{ color: "var(--text-secondary, #697386)" }}
                  >
                    Upload and analyze a new call recording
                  </p>
                </div>
              </div>
            </Link>

            <Link
              href="/dashboard/my-reports"
              className="rounded-lg p-6 transition-shadow hover:shadow-md"
              style={cardStyle}
            >
              <div className="flex items-center mb-4">
                <div
                  className="w-12 h-12 rounded-lg flex items-center justify-center"
                  style={{
                    background: "var(--accent-bg, #f0efff)",
                  }}
                >
                  <FileText
                    className="w-6 h-6"
                    style={{ color: "var(--accent, #635bff)" }}
                  />
                </div>
                <div className="ml-4">
                  <p
                    className="text-sm font-semibold"
                    style={{ color: "var(--text-primary, #1a1f36)" }}
                  >
                    My Reports
                  </p>
                  <p
                    className="text-sm"
                    style={{ color: "var(--text-secondary, #697386)" }}
                  >
                    View and analyze all your call recordings
                  </p>
                </div>
              </div>
            </Link>
          </div>

          {/* Stats Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="rounded-lg p-6" style={cardStyle}>
              <div className="mb-4">
                <div
                  className="w-10 h-10 rounded-lg flex items-center justify-center"
                  style={{ background: "var(--accent-bg, #f0efff)" }}
                >
                  <Phone
                    className="w-5 h-5"
                    style={{ color: "var(--accent, #635bff)" }}
                  />
                </div>
              </div>
              <p
                className="text-sm mb-1"
                style={{ color: "var(--text-secondary, #697386)" }}
              >
                Total Calls
              </p>
              <p
                className="text-3xl font-bold"
                style={{ color: "var(--text-primary, #1a1f36)" }}
              >
                {stats.totalCalls}
              </p>
            </div>

            <div className="rounded-lg p-6" style={cardStyle}>
              <div className="mb-4">
                <div
                  className="w-10 h-10 rounded-lg flex items-center justify-center"
                  style={{ background: "var(--accent-bg, #f0efff)" }}
                >
                  <Clock
                    className="w-5 h-5"
                    style={{ color: "var(--accent, #635bff)" }}
                  />
                </div>
              </div>
              <p
                className="text-sm mb-1"
                style={{ color: "var(--text-secondary, #697386)" }}
              >
                Avg Duration
              </p>
              <p
                className="text-3xl font-bold"
                style={{ color: "var(--text-primary, #1a1f36)" }}
              >
                {formatDuration(stats.avgDuration)}
              </p>
            </div>
          </div>
        </div>

        {/* Recent Activity and Latest Report */}
        <div className="space-y-6">
          {/* Recent Calls */}
          <div className="rounded-lg p-6" style={cardStyle}>
            <div className="flex items-center justify-between mb-6">
              <h2
                className="text-base font-semibold flex items-center"
                style={{ color: "var(--text-primary, #1a1f36)" }}
              >
                <Activity
                  className="w-5 h-5 mr-2"
                  style={{ color: "var(--accent, #635bff)" }}
                />
                Recent Calls
              </h2>
              <Link
                href="/dashboard/calls"
                className="text-sm font-medium"
                style={{ color: "var(--accent, #635bff)" }}
              >
                View All
              </Link>
            </div>
            <div className="space-y-3">
              {recentCalls.map((call) => (
                <div
                  key={call.id}
                  className="flex items-center justify-between p-4 rounded-lg transition-colors"
                  style={{
                    background: "var(--background, #f6f8fa)",
                    border: "1px solid var(--border, #e3e8ee)",
                  }}
                >
                  <div className="flex items-center space-x-3">
                    <div
                      className="w-10 h-10 rounded-full flex items-center justify-center"
                      style={{ background: "var(--accent-bg, #f0efff)" }}
                    >
                      <FileText
                        className="w-4 h-4"
                        style={{ color: "var(--accent, #635bff)" }}
                      />
                    </div>
                    <div>
                      <p
                        className="font-medium text-sm"
                        style={{ color: "var(--text-primary, #1a1f36)" }}
                      >
                        Call #{call.id}
                      </p>
                      <p
                        className="text-xs"
                        style={{ color: "var(--text-secondary, #697386)" }}
                      >
                        {call.date}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p
                      className="font-medium text-sm"
                      style={{ color: "var(--text-primary, #1a1f36)" }}
                    >
                      {formatDuration(call.duration)}
                    </p>
                    <p
                      className="text-xs font-medium"
                      style={{ color: "#0d9488" }}
                    >
                      {call.score}% score
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Latest Performance Report */}
          <div className="rounded-lg p-6" style={cardStyle}>
            <div className="flex items-center justify-between mb-6">
              <h2
                className="text-base font-semibold flex items-center"
                style={{ color: "var(--text-primary, #1a1f36)" }}
              >
                <Target
                  className="w-5 h-5 mr-2"
                  style={{ color: "var(--accent, #635bff)" }}
                />
                Latest Report
              </h2>
              <Link
                href="/dashboard/reports"
                className="text-sm font-medium"
                style={{ color: "var(--accent, #635bff)" }}
              >
                View All
              </Link>
            </div>
            <div className="space-y-4">
              <div
                className="flex items-center justify-between p-4 rounded-lg"
                style={{
                  background: "var(--accent-bg, #f0efff)",
                  border: "1px solid var(--border, #e3e8ee)",
                }}
              >
                <div>
                  <p
                    className="font-semibold text-sm"
                    style={{ color: "var(--text-primary, #1a1f36)" }}
                  >
                    {latestReport.type}
                  </p>
                  <p
                    className="text-xs"
                    style={{ color: "var(--text-secondary, #697386)" }}
                  >
                    {latestReport.date}
                  </p>
                </div>
                <div className="text-right">
                  <p
                    className="text-2xl font-bold"
                    style={{ color: "var(--accent, #635bff)" }}
                  >
                    {latestReport.score}%
                  </p>
                  <p
                    className="text-xs font-medium"
                    style={{ color: "var(--text-secondary, #697386)" }}
                  >
                    {latestReport.overallRating}
                  </p>
                </div>
              </div>

              <div>
                <h3
                  className="text-xs font-semibold uppercase tracking-wider mb-3"
                  style={{ color: "var(--text-secondary, #697386)" }}
                >
                  Strengths
                </h3>
                <div className="space-y-2">
                  {latestReport.strengths.map((strength, index) => (
                    <div key={index} className="flex items-start space-x-2">
                      <CheckCircle
                        className="w-4 h-4 mt-0.5 flex-shrink-0"
                        style={{ color: "#0d9488" }}
                      />
                      <p
                        className="text-sm"
                        style={{ color: "var(--text-primary, #1a1f36)" }}
                      >
                        {strength}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3
                  className="text-xs font-semibold uppercase tracking-wider mb-3"
                  style={{ color: "var(--text-secondary, #697386)" }}
                >
                  Areas to Improve
                </h3>
                <div className="space-y-2">
                  {latestReport.improvements.map((improvement, index) => (
                    <div key={index} className="flex items-start space-x-2">
                      <TrendingUp
                        className="w-4 h-4 mt-0.5 flex-shrink-0"
                        style={{ color: "#e77c40" }}
                      />
                      <p
                        className="text-sm"
                        style={{ color: "var(--text-primary, #1a1f36)" }}
                      >
                        {improvement}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
