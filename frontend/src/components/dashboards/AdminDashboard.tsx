"use client";

import { useState, useEffect } from "react";
import { Building2, Users, Activity, Phone } from "lucide-react";
import Link from "next/link";
import { PageHeader, StatsCard } from "@/components/ui";
import { usersApi, companiesApi, callsApi } from "@/lib/api";

interface SystemStats {
  totalCompanies: number;
  totalUsers: number;
  totalAgents: number;
  totalHeads: number;
  totalCalls: number;
}

export default function AdminDashboard() {
  const [stats, setStats] = useState<SystemStats>({
    totalCompanies: 0,
    totalUsers: 0,
    totalAgents: 0,
    totalHeads: 0,
    totalCalls: 0,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        setLoading(true);

        const [companiesRes, usersRes, callsRes] = await Promise.all([
          companiesApi.getAllCompanies(),
          usersApi.getAllUsers(),
          callsApi.getMyCalls(),
        ]);

        const companies = companiesRes.data;
        const users = usersRes.data;
        const calls = callsRes.data;

        const agents = users.filter((u: any) => u.role === "agent").length;
        const heads = users.filter(
          (u: any) => u.role === "head_of_department",
        ).length;

        setStats({
          totalCompanies: Array.isArray(companies) ? companies.length : 0,
          totalUsers: Array.isArray(users) ? users.length : 0,
          totalAgents: agents,
          totalHeads: heads,
          totalCalls: Array.isArray(calls) ? calls.length : 0,
        });
      } catch (error) {
        console.error("Failed to fetch stats:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
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
        title="System Overview"
        subtitle="Manage companies, users, and system-wide analytics"
      />

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
        <StatsCard
          icon={Building2}
          label="Companies"
          value={stats.totalCompanies}
        />
        <StatsCard icon={Users} label="Total Users" value={stats.totalUsers} />
        <StatsCard icon={Users} label="Agents" value={stats.totalAgents} />
        <StatsCard icon={Users} label="Heads" value={stats.totalHeads} />
        <StatsCard icon={Phone} label="Total Calls" value={stats.totalCalls} />
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Link
          href="/dashboard/companies"
          className="rounded-lg p-6 hover:shadow-md transition-shadow"
          style={cardStyle}
        >
          <div className="flex items-center">
            <div
              className="w-12 h-12 rounded-lg flex items-center justify-center"
              style={{ background: "var(--accent-bg)" }}
            >
              <Building2
                className="w-6 h-6"
                style={{ color: "var(--accent)" }}
              />
            </div>
            <div className="ml-4">
              <p
                className="text-sm font-medium"
                style={{ color: "var(--text-primary)" }}
              >
                Companies
              </p>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                View and manage all registered companies
              </p>
            </div>
          </div>
        </Link>

        <Link
          href="/dashboard/users"
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
                Users
              </p>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                Manage system users and permissions
              </p>
            </div>
          </div>
        </Link>
      </div>

      {/* System Activity */}
      <div className="rounded-lg p-6" style={cardStyle}>
        <h2
          className="text-lg font-semibold mb-6 flex items-center"
          style={{ color: "var(--text-primary)" }}
        >
          <Activity
            className="w-5 h-5 mr-2"
            style={{ color: "var(--success)" }}
          />
          Recent System Activity
        </h2>
        <div className="space-y-3">
          {[
            {
              type: "user",
              message: "New agent added to Tech Solutions Inc",
              time: "10 minutes ago",
            },
            {
              type: "company",
              message: "Global Retail Corp updated company settings",
              time: "1 hour ago",
            },
            {
              type: "call",
              message: "348 calls processed today",
              time: "2 hours ago",
            },
          ].map((activity, index) => (
            <div
              key={index}
              className="flex items-start space-x-3 p-4 rounded-lg transition-colors"
              style={{ background: "var(--background)" }}
            >
              <div
                className="w-2 h-2 rounded-full mt-2"
                style={{ background: "var(--success)" }}
              ></div>
              <div className="flex-1">
                <p
                  className="text-sm mb-1"
                  style={{ color: "var(--text-primary)" }}
                >
                  {activity.message}
                </p>
                <p
                  className="text-xs"
                  style={{ color: "var(--text-tertiary)" }}
                >
                  {activity.time}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
