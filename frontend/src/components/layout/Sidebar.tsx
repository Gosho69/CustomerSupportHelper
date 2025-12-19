"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Phone,
  FileText,
  Users,
  Building2,
  User,
  Settings,
  LogOut,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { useState } from "react";
import { useAuthStore } from "@/store/authStore";

interface SidebarProps {
  userRole: "admin" | "head_of_department" | "agent";
}

export default function Sidebar({ userRole }: SidebarProps) {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const { clearAuth } = useAuthStore();

  const handleLogout = () => {
    // clearAuth(); // Commented out for demo
    window.location.href = "/login";
  };

  const navItems = {
    admin: [
      { href: "/dashboard", label: "Overview", icon: LayoutDashboard },
      { href: "/dashboard/companies", label: "Companies", icon: Building2 },
      { href: "/dashboard/users", label: "Users", icon: Users },
      { href: "/dashboard/reports", label: "Reports", icon: FileText },
      { href: "/dashboard/profile", label: "Profile", icon: User },
    ],
    head_of_department: [
      { href: "/dashboard", label: "Overview", icon: LayoutDashboard },
      { href: "/dashboard/team", label: "My Team", icon: Users },
      { href: "/dashboard/reports", label: "Reports", icon: FileText },
      { href: "/dashboard/calls", label: "All Calls", icon: Phone },
      { href: "/dashboard/profile", label: "Profile", icon: User },
    ],
    agent: [
      { href: "/dashboard", label: "Overview", icon: LayoutDashboard },
      { href: "/dashboard/calls", label: "My Calls", icon: Phone },
      { href: "/dashboard/upload-call", label: "Upload Call", icon: Phone },
      { href: "/dashboard/my-reports", label: "My Reports", icon: FileText },
      { href: "/dashboard/profile", label: "Profile", icon: User },
    ],
  };

  const items = navItems[userRole];

  return (
    <aside
      className={`${
        collapsed ? "w-20" : "w-64"
      } bg-slate-900/50 backdrop-blur-md border-r border-white/10 transition-all duration-300 flex flex-col h-screen`}
    >
      {/* Sidebar Header */}
      <div className="p-4 border-b border-white/10 flex items-center justify-between flex-shrink-0">
        {!collapsed && (
          <h2 className="text-white font-semibold text-lg">Dashboard</h2>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-2 hover:bg-white/10 rounded-lg transition-colors"
        >
          {collapsed ? (
            <ChevronRight className="w-5 h-5 text-gray-400" />
          ) : (
            <ChevronLeft className="w-5 h-5 text-gray-400" />
          )}
        </button>
      </div>

      {/* Navigation Items */}
      <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
        {items.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href;

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                isActive
                  ? "bg-gradient-to-r from-blue-600 to-cyan-600 text-white shadow-lg shadow-blue-500/30"
                  : "text-gray-400 hover:text-white hover:bg-white/10"
              }`}
            >
              <Icon className="w-5 h-5 flex-shrink-0" />
              {!collapsed && <span className="font-medium">{item.label}</span>}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
