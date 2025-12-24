"use client";

import { useState } from "react";
import { Building2, Users, Activity, Phone } from "lucide-react";
import Link from "next/link";
import { StatsCard } from "@/components/ui";

interface SystemStats {
  totalCompanies: number;
  totalUsers: number;
  totalAgents: number;
  totalHeads: number;
  totalCalls: number;
}

export default function AdminDashboard() {
  const [stats, setStats] = useState<SystemStats>({
    totalCompanies: 12,
    totalUsers: 89,
    totalAgents: 64,
    totalHeads: 12,
    totalCalls: 1247,
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white">System Overview</h1>
        <p className="text-gray-400 mt-1">
          Manage companies, users, and system-wide analytics
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
        <StatsCard
          icon={Building2}
          iconColor="bg-indigo-500/20 text-indigo-400"
          label="Companies"
          value={stats.totalCompanies}
        />
        <StatsCard
          icon={Users}
          iconColor="bg-blue-500/20 text-blue-400"
          label="Total Users"
          value={stats.totalUsers}
        />
        <StatsCard
          icon={Users}
          iconColor="bg-green-500/20 text-green-400"
          label="Agents"
          value={stats.totalAgents}
        />
        <StatsCard
          icon={Users}
          iconColor="bg-purple-500/20 text-purple-400"
          label="Heads"
          value={stats.totalHeads}
        />
        <StatsCard
          icon={Phone}
          iconColor="bg-cyan-500/20 text-cyan-400"
          label="Total Calls"
          value={stats.totalCalls}
        />
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Link
          href="/dashboard/companies"
          className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6 hover:bg-slate-800/70 transition-all group"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-indigo-500/20 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
              <Building2 className="w-6 h-6 text-indigo-400" />
            </div>
            <span className="text-indigo-400 text-sm font-medium">Manage</span>
          </div>
          <h3 className="text-lg font-semibold text-white mb-2">Companies</h3>
          <p className="text-gray-400 text-sm">
            View and manage all registered companies
          </p>
        </Link>

        <Link
          href="/dashboard/users"
          className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6 hover:bg-slate-800/70 transition-all group"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
              <Users className="w-6 h-6 text-blue-400" />
            </div>
            <span className="text-blue-400 text-sm font-medium">Manage</span>
          </div>
          <h3 className="text-lg font-semibold text-white mb-2">Users</h3>
          <p className="text-gray-400 text-sm">
            Manage system users and permissions
          </p>
        </Link>
      </div>

      {/* System Activity */}
      <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
        <h2 className="text-xl font-bold text-white mb-6 flex items-center">
          <Activity className="w-5 h-5 mr-2 text-green-400" />
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
              className="flex items-start space-x-3 p-4 bg-slate-900/50 rounded-lg hover:bg-slate-900/70 transition-colors"
            >
              <div className="w-2 h-2 bg-green-400 rounded-full mt-2"></div>
              <div className="flex-1">
                <p className="text-white text-sm mb-1">{activity.message}</p>
                <p className="text-gray-400 text-xs">{activity.time}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
