"use client";

import { Bell, Search, User, LogOut } from "lucide-react";
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
  const profileMenuRef = useRef<HTMLDivElement | null>(null);
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

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        showProfileMenu &&
        profileMenuRef.current &&
        profileButtonRef.current &&
        !profileMenuRef.current.contains(event.target as Node) &&
        !profileButtonRef.current.contains(event.target as Node)
      ) {
        setShowProfileMenu(false);
      }
    };

    if (showProfileMenu) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [showProfileMenu]);

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
                    ref={profileMenuRef}
                    className="w-72 bg-gradient-to-b from-slate-800 to-slate-900 border border-white/20 rounded-xl shadow-2xl py-2 backdrop-blur-md"
                    style={{
                      position: "fixed",
                      top: coords.top,
                      left: Math.max(8, coords.left - 288),
                      zIndex: 99999,
                      overflow: "visible",
                    }}
                  >
                    {/* User Info Header */}
                    <div className="px-5 py-4 border-b border-white/10">
                      <div className="flex items-center space-x-3 mb-3">
                        <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-full flex items-center justify-center flex-shrink-0">
                          <User className="w-6 h-6 text-white" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-base font-semibold text-white truncate">
                            {user?.first_name} {user?.last_name}
                          </p>
                          <p className="text-xs text-gray-400 truncate">
                            {maskEmail(user?.email)}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="inline-flex items-center px-3 py-1 text-xs font-semibold bg-gradient-to-r from-blue-600/20 to-cyan-600/20 text-blue-300 rounded-full border border-blue-500/30">
                          {formatRoleLabel(String(user?.role || ""))}
                        </span>
                      </div>
                    </div>

                    {/* Profile Settings Link */}
                    <div className="py-2">
                      <a
                        href="/dashboard/profile"
                        className="flex items-center px-5 py-2.5 text-sm text-gray-300 hover:bg-white/10 hover:text-white transition-colors"
                      >
                        <User className="w-4 h-4 mr-3 text-gray-400" />
                        Profile Settings
                      </a>
                    </div>

                    {/* Demo Role Switcher */}
                    <div className="px-5 py-3 border-t border-white/10">
                      <p className="text-xs font-medium text-gray-400 mb-3">
                        Switch demo role
                      </p>
                      <div className="grid grid-cols-3 gap-2">
                        <button
                          onClick={() => switchRole("agent")}
                          className={`py-2 px-3 text-xs font-medium rounded-lg transition-all ${
                            userRole === "agent"
                              ? "bg-gradient-to-r from-blue-600 to-cyan-600 text-white shadow-lg"
                              : "bg-slate-700/50 text-gray-300 hover:bg-slate-600"
                          }`}
                        >
                          Agent
                        </button>
                        <button
                          onClick={() => switchRole("head_of_department")}
                          className={`py-2 px-3 text-xs font-medium rounded-lg transition-all ${
                            userRole === "head_of_department"
                              ? "bg-gradient-to-r from-blue-600 to-cyan-600 text-white shadow-lg"
                              : "bg-slate-700/50 text-gray-300 hover:bg-slate-600"
                          }`}
                        >
                          Head
                        </button>
                        <button
                          onClick={() => switchRole("admin")}
                          className={`py-2 px-3 text-xs font-medium rounded-lg transition-all ${
                            userRole === "admin"
                              ? "bg-gradient-to-r from-blue-600 to-cyan-600 text-white shadow-lg"
                              : "bg-slate-700/50 text-gray-300 hover:bg-slate-600"
                          }`}
                        >
                          Admin
                        </button>
                      </div>
                    </div>

                    {/* Logout Button */}
                    <div className="border-t border-white/10 pt-2 pb-2 px-2">
                      <button
                        onClick={() => {
                          setShowProfileMenu(false);
                          setShowLogoutConfirm(true);
                        }}
                        className="flex items-center w-full px-4 py-2.5 text-sm text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
                      >
                        <LogOut className="w-4 h-4 mr-3" />
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
