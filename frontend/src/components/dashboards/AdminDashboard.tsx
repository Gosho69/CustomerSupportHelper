"use client";

import { useEffect, useState } from "react";
import {
  Building2,
  Users,
  FileText,
  Activity,
  TrendingUp,
  UserPlus,
  Building,
  Shield,
  BarChart3,
  Phone,
} from "lucide-react";
import Link from "next/link";

interface SystemStats {
  totalCompanies: number;
  totalUsers: number;
  totalAgents: number;
  totalHeads: number;
  totalCalls: number;
  totalReports: number;
  activeUsers: number;
  systemHealth: number;
}

interface Company {
  id: number;
  name: string;
  industry: string;
  employees: number;
  head: string;
  status: "active" | "inactive";
}

interface RecentActivity {
  id: number;
  type: "user" | "company" | "report" | "call";
  message: string;
  time: string;
}

export default function AdminDashboard() {
  const [stats, setStats] = useState<SystemStats>({
    totalCompanies: 12,
    totalUsers: 89,
    totalAgents: 64,
    totalHeads: 12,
    totalCalls: 1247,
    totalReports: 156,
    activeUsers: 73,
    systemHealth: 98,
  });

  const [companies, setCompanies] = useState<Company[]>([
    {
      id: 1,
      name: "Tech Solutions Inc",
      industry: "Technology",
      employees: 15,
      head: "John Manager",
      status: "active",
    },
    {
      id: 2,
      name: "Global Retail Corp",
      industry: "Retail",
      employees: 23,
      head: "Sarah Director",
      status: "active",
    },
    {
      id: 3,
      name: "Finance Plus",
      industry: "Finance",
      employees: 12,
      head: "Mike Head",
      status: "active",
    },
    {
      id: 4,
      name: "Health Services",
      industry: "Healthcare",
      employees: 18,
      head: "Emily Lead",
      status: "active",
    },
  ]);

  const [recentActivity, setRecentActivity] = useState<RecentActivity[]>([
    {
      id: 1,
      type: "user",
      message: "New agent added to Tech Solutions Inc",
      time: "10 minutes ago",
    },
    {
      id: 2,
      type: "company",
      message: "Global Retail Corp updated company settings",
      time: "1 hour ago",
    },
    {
      id: 3,
      type: "report",
      message: "15 new performance reports generated",
      time: "2 hours ago",
    },
    {
      id: 4,
      type: "user",
      message: "New head of department created",
      time: "3 hours ago",
    },
  ]);

  const [topPerformers, setTopPerformers] = useState([
    { name: "Tech Solutions Inc", score: 94, calls: 345 },
    { name: "Finance Plus", score: 91, calls: 289 },
    { name: "Health Services", score: 88, calls: 312 },
  ]);

  const getActivityIcon = (type: string) => {
    switch (type) {
      case "user":
        return <UserPlus className="w-5 h-5 text-blue-400" />;
      case "company":
        return <Building2 className="w-5 h-5 text-purple-400" />;
      case "report":
        return <FileText className="w-5 h-5 text-green-400" />;
      case "call":
        return <Phone className="w-5 h-5 text-cyan-400" />;
      default:
        return <Activity className="w-5 h-5 text-gray-400" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Welcome Section */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl p-8 text-white">
        <h1 className="text-3xl font-bold mb-2">System Overview</h1>
        <p className="text-indigo-100 mb-6">
          Manage companies, users, and system-wide analytics
        </p>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
            <p className="text-indigo-100 text-sm mb-1">Companies</p>
            <p className="text-3xl font-bold">{stats.totalCompanies}</p>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
            <p className="text-indigo-100 text-sm mb-1">Total Users</p>
            <p className="text-3xl font-bold">{stats.totalUsers}</p>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
            <p className="text-indigo-100 text-sm mb-1">Active Users</p>
            <p className="text-3xl font-bold">{stats.activeUsers}</p>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
            <p className="text-indigo-100 text-sm mb-1">System Health</p>
            <p className="text-3xl font-bold">{stats.systemHealth}%</p>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
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
            Manage all registered companies
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
          <p className="text-gray-400 text-sm">Manage system users</p>
        </Link>

        <Link
          href="/dashboard/reports"
          className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6 hover:bg-slate-800/70 transition-all group"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-purple-500/20 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
              <FileText className="w-6 h-6 text-purple-400" />
            </div>
            <span className="text-purple-400 text-sm font-medium">
              View All
            </span>
          </div>
          <h3 className="text-lg font-semibold text-white mb-2">Reports</h3>
          <p className="text-gray-400 text-sm">View all system reports</p>
        </Link>

        <Link
          href="/dashboard/settings"
          className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6 hover:bg-slate-800/70 transition-all group"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-cyan-500/20 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
              <Shield className="w-6 h-6 text-cyan-400" />
            </div>
            <span className="text-cyan-400 text-sm font-medium">Configure</span>
          </div>
          <h3 className="text-lg font-semibold text-white mb-2">Settings</h3>
          <p className="text-gray-400 text-sm">System configuration</p>
        </Link>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-6 gap-6">
        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="w-10 h-10 bg-indigo-500/20 rounded-lg flex items-center justify-center mb-4">
            <Building2 className="w-5 h-5 text-indigo-400" />
          </div>
          <p className="text-gray-400 text-sm mb-1">Companies</p>
          <p className="text-2xl font-bold text-white">
            {stats.totalCompanies}
          </p>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center mb-4">
            <Users className="w-5 h-5 text-blue-400" />
          </div>
          <p className="text-gray-400 text-sm mb-1">Total Users</p>
          <p className="text-2xl font-bold text-white">{stats.totalUsers}</p>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center mb-4">
            <UserPlus className="w-5 h-5 text-green-400" />
          </div>
          <p className="text-gray-400 text-sm mb-1">Agents</p>
          <p className="text-2xl font-bold text-white">{stats.totalAgents}</p>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center mb-4">
            <Shield className="w-5 h-5 text-purple-400" />
          </div>
          <p className="text-gray-400 text-sm mb-1">Heads</p>
          <p className="text-2xl font-bold text-white">{stats.totalHeads}</p>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="w-10 h-10 bg-cyan-500/20 rounded-lg flex items-center justify-center mb-4">
            <Phone className="w-5 h-5 text-cyan-400" />
          </div>
          <p className="text-gray-400 text-sm mb-1">Total Calls</p>
          <p className="text-2xl font-bold text-white">{stats.totalCalls}</p>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="w-10 h-10 bg-pink-500/20 rounded-lg flex items-center justify-center mb-4">
            <FileText className="w-5 h-5 text-pink-400" />
          </div>
          <p className="text-gray-400 text-sm mb-1">Reports</p>
          <p className="text-2xl font-bold text-white">{stats.totalReports}</p>
        </div>
      </div>

      {/* Companies and Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Companies List */}
        <div className="lg:col-span-2 bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-white flex items-center">
              <Building className="w-5 h-5 mr-2 text-indigo-400" />
              Companies
            </h2>
            <Link
              href="/dashboard/companies"
              className="text-sm text-blue-400 hover:text-blue-300"
            >
              View All
            </Link>
          </div>
          <div className="space-y-4">
            {companies.map((company) => (
              <div
                key={company.id}
                className="flex items-center justify-between p-4 bg-slate-900/50 rounded-lg hover:bg-slate-900/70 transition-colors"
              >
                <div className="flex items-center space-x-4 flex-1">
                  <div className="w-12 h-12 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-lg flex items-center justify-center">
                    <Building2 className="w-6 h-6 text-white" />
                  </div>
                  <div className="flex-1">
                    <p className="text-white font-medium">{company.name}</p>
                    <p className="text-gray-400 text-sm">
                      {company.industry} • {company.employees} employees
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-white text-sm font-medium">
                    {company.head}
                  </p>
                  <span
                    className={`inline-block px-2 py-1 text-xs rounded-full ${
                      company.status === "active"
                        ? "bg-green-500/20 text-green-400"
                        : "bg-gray-500/20 text-gray-400"
                    }`}
                  >
                    {company.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Recent Activity */}
        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <h2 className="text-xl font-bold text-white mb-6 flex items-center">
            <Activity className="w-5 h-5 mr-2 text-green-400" />
            Recent Activity
          </h2>
          <div className="space-y-4">
            {recentActivity.map((activity) => (
              <div
                key={activity.id}
                className="flex items-start space-x-3 p-3 bg-slate-900/50 rounded-lg"
              >
                <div className="w-8 h-8 bg-slate-800 rounded-lg flex items-center justify-center flex-shrink-0">
                  {getActivityIcon(activity.type)}
                </div>
                <div className="flex-1">
                  <p className="text-white text-sm mb-1">{activity.message}</p>
                  <p className="text-gray-400 text-xs">{activity.time}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Top Performers */}
      <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-white flex items-center">
            <BarChart3 className="w-5 h-5 mr-2 text-yellow-400" />
            Top Performing Companies
          </h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {topPerformers.map((performer, index) => (
            <div
              key={index}
              className="p-6 bg-gradient-to-br from-slate-900/50 to-slate-800/50 rounded-lg border border-white/10 relative overflow-hidden"
            >
              <div className="absolute top-2 right-2">
                <span className="flex items-center justify-center w-8 h-8 bg-yellow-500/20 text-yellow-400 rounded-full text-sm font-bold">
                  #{index + 1}
                </span>
              </div>
              <div className="mb-4">
                <TrendingUp className="w-8 h-8 text-green-400" />
              </div>
              <p className="text-white font-semibold mb-2">{performer.name}</p>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-xs">Performance</p>
                  <p className="text-2xl font-bold text-green-400">
                    {performer.score}%
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-gray-400 text-xs">Calls</p>
                  <p className="text-lg font-semibold text-white">
                    {performer.calls}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
