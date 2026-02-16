"use client";

import { useEffect, useState } from "react";
import AgentDashboard from "@/components/dashboards/AgentDashboard";
import HeadOfDepartmentDashboard from "@/components/dashboards/HeadOfDepartmentDashboard";
import AdminDashboard from "@/components/dashboards/AdminDashboard";
import { useAuthStore } from "@/store/authStore";

export default function DashboardPage() {
  const { user } = useAuthStore();
  const [userRole, setUserRole] = useState<string | null>(null);

  useEffect(() => {
    // Get role from auth store
    if (user?.role) {
      setUserRole(user.role);
    }
  }, [user]);

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
