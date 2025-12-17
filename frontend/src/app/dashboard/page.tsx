"use client";

import { useEffect, useState } from "react";
import AgentDashboard from "@/components/dashboards/AgentDashboard";
import HeadOfDepartmentDashboard from "@/components/dashboards/HeadOfDepartmentDashboard";
import AdminDashboard from "@/components/dashboards/AdminDashboard";

export default function DashboardPage() {
  const [userRole, setUserRole] = useState<string | null>(null);

  useEffect(() => {
    // Get role from localStorage (demo mode)
    const storedRole = localStorage.getItem("demo_role");
    setUserRole(storedRole || "agent");
  }, []);

  // Don't render anything until role is determined (prevents hydration mismatch)
  if (!userRole) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-400">Loading dashboard...</div>
      </div>
    );
  }

  // Render appropriate dashboard based on role
  if (userRole === "admin") {
    return <AdminDashboard />;
  }

  if (userRole === "head_of_department") {
    return <HeadOfDepartmentDashboard />;
  }

  return <AgentDashboard />;
}
