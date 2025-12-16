"use client";

import {
  Phone,
  TrendingUp,
  Users,
  FileText,
  ArrowUp,
  ArrowDown,
} from "lucide-react";

export default function DashboardPage() {
  const user = {
    first_name: "Demo",
    username: "demo_user",
    role: "agent" as const,
  };

  const stats = {
    agent: [
      {
        label: "Total Calls",
        value: "142",
        change: "+12%",
        trend: "up",
        icon: Phone,
      },
      {
        label: "Avg. Score",
        value: "8.5",
        change: "+0.3",
        trend: "up",
        icon: TrendingUp,
      },
      {
        label: "This Week",
        value: "23",
        change: "+5",
        trend: "up",
        icon: Phone,
      },
      {
        label: "Reports",
        value: "8",
        change: "2 new",
        trend: "up",
        icon: FileText,
      },
    ],
    head_of_department: [
      {
        label: "Team Members",
        value: "12",
        change: "+2",
        trend: "up",
        icon: Users,
      },
      {
        label: "Total Calls",
        value: "1,247",
        change: "+18%",
        trend: "up",
        icon: Phone,
      },
      {
        label: "Avg. Team Score",
        value: "8.2",
        change: "+0.5",
        trend: "up",
        icon: TrendingUp,
      },
      {
        label: "Reports Generated",
        value: "45",
        change: "12 this week",
        trend: "up",
        icon: FileText,
      },
    ],
    admin: [
      {
        label: "Total Companies",
        value: "8",
        change: "+1",
        trend: "up",
        icon: Users,
      },
      {
        label: "Total Users",
        value: "156",
        change: "+8",
        trend: "up",
        icon: Users,
      },
      {
        label: "Calls Analyzed",
        value: "12.5K",
        change: "+23%",
        trend: "up",
        icon: Phone,
      },
      {
        label: "System Health",
        value: "98%",
        change: "+2%",
        trend: "up",
        icon: TrendingUp,
      },
    ],
  };

  const currentStats = user?.role ? stats[user.role] : stats.agent;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">
          Welcome back, {user?.first_name || user?.username}! 👋
        </h1>
        <p className="text-gray-400">
          Here's what's happening with your{" "}
          {user?.role === "agent"
            ? "performance"
            : user?.role === "head_of_department"
            ? "team"
            : "platform"}{" "}
          today.
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {currentStats.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <div
              key={index}
              className="bg-slate-800/50 backdrop-blur-sm border border-white/10 rounded-xl p-6 hover:border-blue-500/30 transition-all duration-200"
            >
              <div className="flex items-center justify-between mb-4">
                <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center">
                  <Icon className="w-6 h-6 text-blue-400" />
                </div>
                <div
                  className={`flex items-center space-x-1 text-sm ${
                    stat.trend === "up" ? "text-green-400" : "text-red-400"
                  }`}
                >
                  {stat.trend === "up" ? (
                    <ArrowUp className="w-4 h-4" />
                  ) : (
                    <ArrowDown className="w-4 h-4" />
                  )}
                  <span>{stat.change}</span>
                </div>
              </div>
              <h3 className="text-3xl font-bold text-white mb-1">
                {stat.value}
              </h3>
              <p className="text-gray-400 text-sm">{stat.label}</p>
            </div>
          );
        })}
      </div>

      {/* Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Calls */}
        <div className="bg-slate-800/50 backdrop-blur-sm border border-white/10 rounded-xl p-6">
          <h3 className="text-xl font-bold text-white mb-4">Recent Calls</h3>
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="flex items-center justify-between p-4 bg-slate-900/50 rounded-lg hover:bg-slate-900 transition-colors cursor-pointer"
              >
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
                    <Phone className="w-5 h-5 text-blue-400" />
                  </div>
                  <div>
                    <p className="text-white font-medium">Call #{1000 + i}</p>
                    <p className="text-gray-400 text-sm">2 hours ago</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-green-400 font-semibold">8.5/10</p>
                  <p className="text-gray-400 text-sm">Score</p>
                </div>
              </div>
            ))}
          </div>
          <button className="w-full mt-4 py-2 text-blue-400 hover:text-blue-300 font-medium transition-colors">
            View all calls →
          </button>
        </div>

        {/* Recent Reports */}
        <div className="bg-slate-800/50 backdrop-blur-sm border border-white/10 rounded-xl p-6">
          <h3 className="text-xl font-bold text-white mb-4">Recent Reports</h3>
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="flex items-center justify-between p-4 bg-slate-900/50 rounded-lg hover:bg-slate-900 transition-colors cursor-pointer"
              >
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center">
                    <FileText className="w-5 h-5 text-purple-400" />
                  </div>
                  <div>
                    <p className="text-white font-medium">
                      {i === 1 ? "Weekly" : "Monthly"} Report
                    </p>
                    <p className="text-gray-400 text-sm">
                      {i === 1 ? "This week" : `Week ${i - 1}`}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-blue-400 font-semibold">View</p>
                </div>
              </div>
            ))}
          </div>
          <button className="w-full mt-4 py-2 text-blue-400 hover:text-blue-300 font-medium transition-colors">
            View all reports →
          </button>
        </div>
      </div>
    </div>
  );
}
