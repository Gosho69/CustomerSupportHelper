"use client";

import { useEffect, useState } from "react";
import { Phone, FileText, Clock, Upload } from "lucide-react";
import Link from "next/link";
import { PageHeader } from "@/components/ui";
import { callsApi, reportsApi } from "@/lib/api";

interface DashboardStats {
  totalCalls: number;
  avgDuration: number;
  totalReports: number;
}

export default function AgentDashboard() {
  const [stats, setStats] = useState<DashboardStats>({
    totalCalls: 0,
    avgDuration: 0,
    totalReports: 0,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [callsRes, reportsRes] = await Promise.all([
          callsApi.getMyCalls(),
          reportsApi.getMyReports(),
        ]);

        const calls = callsRes.data || [];
        const reports = reportsRes.data || [];

        const totalCalls = calls.length;
        const avgDuration =
          totalCalls > 0
            ? Math.round(
                calls.reduce(
                  (sum: number, c: any) => sum + (c.duration || 0),
                  0,
                ) / totalCalls,
              )
            : 0;

        setStats({
          totalCalls,
          avgDuration,
          totalReports: reports.length,
        });
      } catch (error) {
        console.error("Failed to fetch dashboard data:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

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

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div style={{ color: "var(--text-secondary)" }}>
          Loading dashboard...
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <PageHeader
          title={`Welcome back, Agent!`}
          subtitle={`Here's your performance overview`}
        />
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
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
        <div className="rounded-lg p-6" style={cardStyle}>
          <div className="mb-4">
            <div
              className="w-10 h-10 rounded-lg flex items-center justify-center"
              style={{ background: "var(--accent-bg, #f0efff)" }}
            >
              <FileText
                className="w-5 h-5"
                style={{ color: "var(--accent, #635bff)" }}
              />
            </div>
          </div>
          <p
            className="text-sm mb-1"
            style={{ color: "var(--text-secondary, #697386)" }}
          >
            Total Reports
          </p>
          <p
            className="text-3xl font-bold"
            style={{ color: "var(--text-primary, #1a1f36)" }}
          >
            {stats.totalReports}
          </p>
        </div>
      </div>

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
              style={{ background: "var(--accent-bg, #f0efff)" }}
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
              style={{ background: "var(--accent-bg, #f0efff)" }}
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
                View and analyze all your performance reports
              </p>
            </div>
          </div>
        </Link>
      </div>
    </div>
  );
}
