"use client";

import { useEffect, useState } from "react";
import { Users, FileText, Phone, Award } from "lucide-react";
import Link from "next/link";
import { PageHeader, StatsCard } from "@/components/ui";
import { usersApi, callsApi } from "@/lib/api";
import { useAuthStore } from "@/store/authStore";

interface DashboardStats {
  totalTeamMembers: number;
  totalTeamCalls: number;
  avgTeamScore: number | null;
}

export default function HeadOfDepartmentDashboard() {
  const { user } = useAuthStore();
  const [stats, setStats] = useState<DashboardStats>({
    totalTeamMembers: 0,
    totalTeamCalls: 0,
    avgTeamScore: null,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [subordinatesRes, callsRes] = await Promise.all([
          usersApi.getSubordinates(),
          callsApi.getMyCalls(), // For HoD, this returns all subordinates' calls
        ]);

        const subordinates = subordinatesRes.data || [];
        const calls = callsRes.data || [];

        const scoredCalls = calls.filter(
          (c: any) =>
            c.behavioral_score != null && !isNaN(Number(c.behavioral_score)),
        );
        const avgTeamScore =
          scoredCalls.length > 0
            ? Math.round(
                scoredCalls.reduce(
                  (sum: number, c: any) => sum + Number(c.behavioral_score),
                  0,
                ) / scoredCalls.length,
              )
            : null;

        setStats({
          totalTeamMembers: subordinates.length,
          totalTeamCalls: calls.length,
          avgTeamScore,
        });
      } catch (error) {
        console.error("Failed to fetch dashboard data:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

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
      <PageHeader
        title={`Welcome back, ${user?.first_name || user?.username || "Manager"}!`}
        subtitle="Monitor and manage your team's performance"
      />

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatsCard
          icon={Users}
          label="Team Members"
          value={stats.totalTeamMembers}
        />
        <StatsCard
          icon={Phone}
          label="Total Calls"
          value={stats.totalTeamCalls}
        />
        <StatsCard
          icon={Award}
          label="Avg Team Score"
          value={
            stats.avgTeamScore != null ? `${stats.avgTeamScore}/100` : "N/A"
          }
        />
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
    </div>
  );
}
