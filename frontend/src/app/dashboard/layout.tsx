"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { Sidebar, DashboardNavbar } from "@/components/layout";
import { useAuthStore } from "@/store/authStore";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const { user, isHydrating, hydrateFromServer } = useAuthStore();

  // Rehydrate user state from server on mount (cookie is sent automatically)
  useEffect(() => {
    if (!user) {
      hydrateFromServer().then(() => {
        const { user: hydratedUser } = useAuthStore.getState();
        if (!hydratedUser) {
          router.push("/login");
        }
      });
    }
  }, []);

  // Show loading while hydrating
  if (isHydrating || !user) {
    return (
      <div
        className="h-screen flex items-center justify-center"
        style={{ background: "var(--background)" }}
      >
        <div className="text-white text-lg">Loading...</div>
      </div>
    );
  }

  const userRole = user.role || "agent";

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
