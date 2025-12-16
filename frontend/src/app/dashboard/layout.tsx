"use client";

import { useEffect, useState } from "react";
import Sidebar from "@/components/Sidebar";
import DashboardNavbar from "@/components/DashboardNavbar";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // Start with a deterministic default so server and initial client render match.
  const [role, setRole] = useState<"agent" | "head_of_department" | "admin">(
    "agent"
  );

  // Read demo role from localStorage only after mount to avoid hydration mismatches.
  useEffect(() => {
    try {
      const stored = localStorage.getItem("demo_role");
      if (stored === "head_of_department" || stored === "admin") {
        setRole(stored as "head_of_department" | "admin");
      }
    } catch (e) {
      // ignore
    }
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 flex">
      {/* Sidebar */}
      <Sidebar userRole={role} />

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Navbar */}
        <DashboardNavbar />

        {/* Page Content */}
        <main className="flex-1 overflow-y-auto p-6">
          <div className="max-w-7xl mx-auto">{children}</div>
        </main>
      </div>
    </div>
  );
}
