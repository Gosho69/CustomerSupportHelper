"use client";

import { useEffect, useState } from "react";
import { AllCalls } from "@/components/calls";
import AgentCallsView from "@/components/calls/agent/AgentCallsView";
import { useAuthStore } from "@/store/authStore";

export default function CallsPage() {
  const { user } = useAuthStore();
  const [userRole, setUserRole] = useState<
    "agent" | "head_of_department" | "admin"
  >("agent");

  useEffect(() => {
    // Get user role from auth store
    if (user?.role) {
      setUserRole(user.role as "agent" | "head_of_department" | "admin");
    }
  }, [user]);

  // If user is head of department or admin, show the AllCalls component
  if (userRole === "head_of_department" || userRole === "admin") {
    return <AllCalls />;
  }

  // For agents, show the existing agent calls view
  return <AgentCallsView />;
}
