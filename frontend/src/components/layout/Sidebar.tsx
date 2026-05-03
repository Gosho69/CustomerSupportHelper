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
  Upload,
  ChevronLeft,
  ChevronRight,
  PhoneCall,
} from "lucide-react";
import { useState } from "react";
import { useAuthStore } from "@/store/authStore";

interface SidebarProps {
  userRole: "admin" | "head_of_department" | "agent";
}

export default function Sidebar({ userRole }: SidebarProps) {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);

  const navItems = {
    admin: [
      { href: "/dashboard", label: "Overview", icon: LayoutDashboard },
      { href: "/dashboard/companies", label: "Companies", icon: Building2 },
      { href: "/dashboard/users", label: "Users", icon: Users },
      { href: "/dashboard/call-center", label: "Call Center", icon: PhoneCall },
      { href: "/dashboard/profile", label: "Profile", icon: User },
    ],
    head_of_department: [
      { href: "/dashboard", label: "Overview", icon: LayoutDashboard },
      { href: "/dashboard/team", label: "My Team", icon: Users },
      { href: "/dashboard/reports", label: "Reports", icon: FileText },
      { href: "/dashboard/calls", label: "All Calls", icon: Phone },
      { href: "/dashboard/call-center", label: "Call Center", icon: PhoneCall },
      { href: "/dashboard/profile", label: "Profile", icon: User },
    ],
    agent: [
      { href: "/dashboard", label: "Overview", icon: LayoutDashboard },
      { href: "/dashboard/calls", label: "My Calls", icon: Phone },
      { href: "/dashboard/upload-call", label: "Upload Call", icon: Upload },
      { href: "/dashboard/call-center", label: "Call Center", icon: PhoneCall },
      { href: "/dashboard/my-reports", label: "My Reports", icon: FileText },
      { href: "/dashboard/profile", label: "Profile", icon: User },
    ],
  };

  const items = navItems[userRole];

  return (
    <aside
      className={`${collapsed ? "w-[68px]" : "w-60"} transition-all duration-200 flex flex-col h-screen bg-white border-r`}
      style={{ borderColor: "var(--border)" }}
    >
      {/* Logo */}
      <div
        className="h-16 flex items-center px-4 border-b"
        style={{ borderColor: "var(--border)" }}
      >
        <Link href="/dashboard" className="flex items-center space-x-2.5">
          <img
            src="/logo-icon.svg"
            alt="AgentSights"
            className="w-8 h-8 flex-shrink-0"
          />
          {!collapsed && (
            <span className="text-[15px] font-semibold">
              <span style={{ color: "var(--text-primary)" }}>Agent</span>
              <span style={{
                background: "linear-gradient(90deg, #4BA8FF, #8B7CF8)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundClip: "text",
              }}>Sights</span>
            </span>
          )}
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
        {items.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href;

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center space-x-2.5 px-3 py-2 rounded-md text-[14px] font-medium transition-colors duration-150 ${
                isActive ? "text-[var(--accent)]" : "hover:bg-[var(--hover-bg)]"
              }`}
              style={{
                color: isActive ? "var(--accent)" : "var(--text-secondary)",
                background: isActive ? "var(--accent-bg)" : undefined,
              }}
            >
              <Icon className="w-[18px] h-[18px] flex-shrink-0" />
              {!collapsed && <span>{item.label}</span>}
            </Link>
          );
        })}
      </nav>

      {/* Collapse toggle */}
      <div
        className="px-3 py-3 border-t"
        style={{ borderColor: "var(--border)" }}
      >
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="flex items-center justify-center w-full p-2 rounded-md hover:bg-[var(--hover-bg)] transition-colors"
          style={{ color: "var(--text-tertiary)" }}
        >
          {collapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <ChevronLeft className="w-4 h-4" />
          )}
        </button>
      </div>
    </aside>
  );
}
