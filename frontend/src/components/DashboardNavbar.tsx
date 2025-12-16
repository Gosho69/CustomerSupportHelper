"use client";

import { Bell, Search, User } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { createPortal } from "react-dom";

export default function DashboardNavbar() {
  // Read role after mount to avoid hydration mismatch
  const [userRole, setUserRole] = useState<
    "agent" | "head_of_department" | "admin"
  >("agent");

  useEffect(() => {
    const stored = localStorage.getItem("demo_role");
    if (stored === "head_of_department" || stored === "admin") {
      setUserRole(stored as "head_of_department" | "admin");
    }
  }, []);

  // Temporarily hardcoded user for demo purposes
  const user = {
    first_name: "Demo",
    last_name: "User",
    username: "demo_user",
    email: "demo@example.com",
    role: userRole,
  } as any;
  const [showProfileMenu, setShowProfileMenu] = useState(false);
  const [showLogoutConfirm, setShowLogoutConfirm] = useState(false);
  const profileButtonRef = useRef<HTMLButtonElement | null>(null);
  const [coords, setCoords] = useState<{ top: number; left: number } | null>(
    null
  );
  const portalRootRef = useRef<HTMLElement | null>(null);

  // Create a portal root div attached to document.body to render dropdowns above all content
  useEffect(() => {
    if (typeof document === "undefined") return;
    const div = document.createElement("div");
    document.body.appendChild(div);
    portalRootRef.current = div;
    return () => {
      if (div.parentNode) div.parentNode.removeChild(div);
      portalRootRef.current = null;
    };
  }, []);

  // Recompute dropdown position on resize/scroll when it's open
  useEffect(() => {
    function update() {
      if (showProfileMenu && profileButtonRef.current) {
        const rect = profileButtonRef.current.getBoundingClientRect();
        setCoords({ top: rect.bottom + 8, left: rect.right });
      }
    }
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    return () => {
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
    };
  }, [showProfileMenu]);

  const switchRole = (role: "agent" | "head_of_department" | "admin") => {
    localStorage.setItem("demo_role", role);
    setUserRole(role);
    setShowProfileMenu(false);
    window.location.reload(); // Reload to update sidebar and other components
  };

  const confirmLogout = () => {
    localStorage.removeItem("demo_role");
    window.location.href = "/login";
  };

  return (
    <header className="bg-slate-900/50 backdrop-blur-md border-b border-white/10 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Search Bar */}
        <div className="flex-1 max-w-xl">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search calls, reports, agents..."
              className="w-full pl-10 pr-4 py-2.5 bg-slate-800/50 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
            />
          </div>
        </div>

        {/* Right Side Actions */}
        <div className="flex items-center space-x-4 ml-6">
          {/* Notifications */}
          <button className="relative p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-all">
            <Bell className="w-5 h-5" />
            <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></span>
          </button>

          {/* User Profile */}
          <div className="relative">
            <button
              ref={profileButtonRef}
              onClick={() => {
                const next = !showProfileMenu;
                setShowProfileMenu(next);
                if (next && profileButtonRef.current) {
                  const rect = profileButtonRef.current.getBoundingClientRect();
                  // position dropdown slightly below the button
                  setCoords({ top: rect.bottom + 8, left: rect.right });
                }
              }}
              className="flex items-center space-x-3 px-3 py-2 hover:bg-white/10 rounded-lg transition-all"
            >
              <div className="w-9 h-9 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-full flex items-center justify-center">
                <User className="w-5 h-5 text-white" />
              </div>
              <div className="text-left">
                <p className="text-sm font-medium text-white">
                  {user?.first_name || user?.username} {user?.last_name || ""}
                </p>
                <p className="text-xs text-gray-400">
                  {formatRoleLabel(user?.role)}
                </p>
              </div>
            </button>

            {/* Profile Dropdown (rendered into portal to avoid clipping) */}
            {showProfileMenu && portalRootRef.current && coords
              ? createPortal(
                  <div
                    className="w-64 bg-slate-800 border border-white/10 rounded-lg shadow-xl py-2"
                    style={{
                      position: "fixed",
                      top: coords.top,
                      left: Math.max(8, coords.left - 256),
                      zIndex: 99999,
                      overflow: "visible",
                    }}
                  >
                    <div className="px-4 py-3 border-b border-white/10">
                      <p className="text-sm font-medium text-white">
                        {user?.first_name} {user?.last_name}
                      </p>
                      <p className="text-xs text-gray-400">
                        {maskEmail(user?.email)}
                      </p>
                      <div className="mt-2">
                        <span className="inline-block px-2 py-0.5 text-xs font-medium bg-slate-700 text-gray-200 rounded">
                          {formatRoleLabel(String(user?.role || ""))}
                        </span>
                      </div>
                    </div>
                    <a
                      href="/dashboard/profile"
                      className="block px-4 py-2 text-sm text-gray-300 hover:bg-white/10 hover:text-white transition-colors"
                    >
                      Profile Settings
                    </a>
                    <a
                      href="/dashboard/settings"
                      className="block px-4 py-2 text-sm text-gray-300 hover:bg-white/10 hover:text-white transition-colors"
                    >
                      Account Settings
                    </a>

                    <div className="px-4 py-2 border-t border-white/10">
                      <p className="text-xs text-gray-400 mb-2">
                        Switch demo role
                      </p>
                      <div className="flex gap-2">
                        <button
                          onClick={() => switchRole("agent")}
                          className="flex-1 py-2 text-sm rounded bg-slate-700 text-white hover:bg-slate-600"
                        >
                          Agent
                        </button>
                        <button
                          onClick={() => switchRole("head_of_department")}
                          className="flex-1 py-2 text-sm rounded bg-slate-700 text-white hover:bg-slate-600"
                        >
                          Head
                        </button>
                        <button
                          onClick={() => switchRole("admin")}
                          className="flex-1 py-2 text-sm rounded bg-slate-700 text-white hover:bg-slate-600"
                        >
                          Admin
                        </button>
                      </div>
                    </div>

                    <div className="border-t border-white/10 mt-2 pt-2 px-4">
                      <button
                        onClick={() => setShowLogoutConfirm(true)}
                        className="block w-full text-left px-4 py-2 text-sm text-red-400 hover:bg-red-500/10 transition-colors"
                      >
                        Logout
                      </button>
                    </div>
                  </div>,
                  portalRootRef.current
                )
              : null}

            {/* Logout Confirmation Modal */}
            {showLogoutConfirm && (
              <div className="fixed inset-0 z-50 flex items-center justify-center">
                <div
                  className="absolute inset-0 bg-black/50"
                  onClick={() => setShowLogoutConfirm(false)}
                />
                <div className="relative bg-slate-800 rounded-lg p-6 w-96 border border-white/10 z-50">
                  <h3 className="text-lg font-semibold text-white mb-2">
                    Confirm Logout
                  </h3>
                  <p className="text-sm text-gray-400 mb-4">
                    Are you sure you want to log out?
                  </p>
                  <div className="flex justify-end gap-2">
                    <button
                      onClick={() => setShowLogoutConfirm(false)}
                      className="px-4 py-2 rounded bg-slate-700 text-gray-200"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={confirmLogout}
                      className="px-4 py-2 rounded bg-red-500 text-white"
                    >
                      Logout
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}

function maskEmail(email: string | undefined) {
  if (!email) return "";
  const [name, domain] = email.split("@");
  const maskedName =
    name.length > 2 ? name[0] + "***" + name[name.length - 1] : name;
  return `${maskedName}@${domain}`;
}

function formatRoleLabel(role: string | undefined) {
  if (!role) return "";
  const label = role.replace("_", " ");
  return label.charAt(0).toUpperCase() + label.slice(1);
}
