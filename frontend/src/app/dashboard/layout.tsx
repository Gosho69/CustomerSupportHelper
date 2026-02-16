"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Sidebar, DashboardNavbar } from "@/components/layout";
import { useAuthStore } from "@/store/authStore";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const { user, isAuthenticated, accessToken } = useAuthStore();
  const [isLoading, setIsLoading] = useState(true);
  const [isMounted, setIsMounted] = useState(false);

  // Wait for client-side hydration
  useEffect(() => {
    setIsMounted(true);
  }, []);

  // Check authentication on mount
  useEffect(() => {
    if (isMounted) {
      // Give a moment for auth store to load from localStorage
      const checkAuth = setTimeout(() => {
        if (!isAuthenticated() || !accessToken) {
          router.push("/login");
        } else {
          setIsLoading(false);
        }
      }, 100);

      return () => clearTimeout(checkAuth);
    }
  }, [isAuthenticated, accessToken, router, isMounted]);

  // Show nothing while checking auth
  if (!isMounted || isLoading || !isAuthenticated()) {
    return (
      <div
        className="h-screen flex items-center justify-center"
        style={{ background: "var(--background)" }}
      >
        <div className="text-white text-lg">Loading...</div>
      </div>
    );
  }

  // Get user role, default to agent if not set
  const userRole = user?.role || "agent";

  return (
    <div
      className="h-screen flex overflow-hidden"
      style={{ background: "var(--background)" }}
    >
      <Sidebar userRole={userRole} />
      <div className="flex-1 flex flex-col overflow-hidden">
        <DashboardNavbar />
        <main
          className="flex-1 overflow-y-auto p-6"
          style={{ background: "var(--background)" }}
        >
          <div className="max-w-7xl mx-auto">{children}</div>
        </main>
      </div>
    </div>
  );
}
