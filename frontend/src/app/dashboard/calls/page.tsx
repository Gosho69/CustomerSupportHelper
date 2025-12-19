"use client";

import { useEffect, useState } from "react";
import { AllCalls } from "@/components/calls";
import AgentCallsView from "@/components/calls/agent/AgentCallsView";

export default function CallsPage() {
  const [userRole, setUserRole] = useState<
    "agent" | "head_of_department" | "admin"
  >("agent");

  useEffect(() => {
    // Get user role from localStorage for demo mode
    const storedRole = localStorage.getItem("demo_role") as
      | "agent"
      | "head_of_department"
      | "admin"
      | null;
    if (storedRole) {
      setUserRole(storedRole);
    }
  }, []);

  // If user is head of department, show the new AllCalls component
  if (userRole === "head_of_department") {
    return <AllCalls />;
  }

  // For agents, show the existing agent calls view
  return <AgentCallsView />;
}
