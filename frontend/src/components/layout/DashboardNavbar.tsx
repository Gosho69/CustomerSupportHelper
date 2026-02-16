"use client";

import { User, LogOut, Settings } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { createPortal } from "react-dom";
import { useAuthStore } from "@/store/authStore";
import { useRouter } from "next/navigation";
import { authApi } from "@/lib/api";

export default function DashboardNavbar() {
  const router = useRouter();
  const { user, clearAuth } = useAuthStore();
  const [showProfileMenu, setShowProfileMenu] = useState(false);
  const [showLogoutConfirm, setShowLogoutConfirm] = useState(false);
  const profileButtonRef = useRef<HTMLButtonElement | null>(null);
  const profileMenuRef = useRef<HTMLDivElement | null>(null);
  const [coords, setCoords] = useState<{ top: number; left: number } | null>(
    null,
  );
  const portalRootRef = useRef<HTMLElement | null>(null);

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
    if (showProfileMenu)
      document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [showProfileMenu]);

  useEffect(() => {
    function update() {
      if (showProfileMenu && profileButtonRef.current) {
        const rect = profileButtonRef.current.getBoundingClientRect();
        setCoords({ top: rect.bottom + 6, left: rect.right });
      }
    }
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    return () => {
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
    };
  }, [showProfileMenu]);

  const confirmLogout = () => {
    authApi.logout();
    clearAuth();
    router.push("/login");
  };

  return (
    <header
      className="h-16 bg-white border-b px-6 flex items-center justify-between"
      style={{ borderColor: "var(--border)" }}
    >
      {/* Spacer */}
      <div className="flex-1" />

      {/* Right */}
      <div className="flex items-center space-x-2 ml-4">
        <div className="relative">
          <button
            ref={profileButtonRef}
            onClick={() => {
              const next = !showProfileMenu;
              setShowProfileMenu(next);
              if (next && profileButtonRef.current) {
                const rect = profileButtonRef.current.getBoundingClientRect();
                setCoords({ top: rect.bottom + 6, left: rect.right });
              }
            }}
            className="flex items-center space-x-2.5 px-2 py-1.5 rounded-md hover:bg-[var(--hover-bg)] transition-colors"
          >
            <div
              className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold text-white"
              style={{ background: "var(--accent)" }}
            >
              {(
                user?.first_name?.[0] ||
                user?.username?.[0] ||
                "U"
              ).toUpperCase()}
            </div>
            <div className="text-left hidden sm:block">
              <p
                className="text-sm font-medium"
                style={{ color: "var(--text-primary)" }}
              >
                {user?.first_name || user?.username} {user?.last_name || ""}
              </p>
              <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
                {formatRoleLabel(user?.role)}
              </p>
            </div>
          </button>

          {showProfileMenu && portalRootRef.current && coords
            ? createPortal(
                <div
                  ref={profileMenuRef}
                  className="w-64 rounded-lg py-1 shadow-lg"
                  style={{
                    position: "fixed",
                    top: coords.top,
                    left: Math.max(8, coords.left - 256),
                    zIndex: 99999,
                    background: "white",
                    border: "1px solid var(--border)",
                  }}
                >
                  <div
                    className="px-4 py-3 border-b"
                    style={{ borderColor: "var(--border)" }}
                  >
                    <div className="flex items-center space-x-3">
                      <div
                        className="w-10 h-10 rounded-full flex items-center justify-center text-sm font-semibold text-white"
                        style={{ background: "var(--accent)" }}
                      >
                        {(user?.first_name?.[0] || "U").toUpperCase()}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p
                          className="text-sm font-semibold truncate"
                          style={{ color: "var(--text-primary)" }}
                        >
                          {user?.first_name} {user?.last_name}
                        </p>
                        <p
                          className="text-xs truncate"
                          style={{ color: "var(--text-tertiary)" }}
                        >
                          {maskEmail(user?.email)}
                        </p>
                      </div>
                    </div>
                    <span
                      className="inline-flex items-center mt-2 px-2.5 py-0.5 text-xs font-medium rounded-full"
                      style={{
                        background: "var(--accent-bg)",
                        color: "var(--accent)",
                      }}
                    >
                      {formatRoleLabel(String(user?.role || ""))}
                    </span>
                  </div>
                  <div className="py-1">
                    <a
                      href="/dashboard/profile"
                      className="flex items-center px-4 py-2 text-sm hover:bg-[var(--hover-bg)] transition-colors"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      <Settings className="w-4 h-4 mr-2.5" />
                      Profile Settings
                    </a>
                  </div>
                  <div
                    className="border-t py-1"
                    style={{ borderColor: "var(--border)" }}
                  >
                    <button
                      onClick={() => {
                        setShowProfileMenu(false);
                        setShowLogoutConfirm(true);
                      }}
                      className="flex items-center w-full px-4 py-2 text-sm hover:bg-[var(--hover-bg)] transition-colors"
                      style={{ color: "var(--danger)" }}
                    >
                      <LogOut className="w-4 h-4 mr-2.5" />
                      Logout
                    </button>
                  </div>
                </div>,
                portalRootRef.current,
              )
            : null}

          {showLogoutConfirm &&
            portalRootRef.current &&
            createPortal(
              <div className="fixed inset-0 z-[99999] flex items-center justify-center p-4">
                <div
                  className="absolute inset-0 bg-black/20 backdrop-blur-[2px]"
                  onClick={() => setShowLogoutConfirm(false)}
                />
                <div
                  className="relative bg-white rounded-xl p-6 w-full max-w-sm shadow-xl"
                  style={{ border: "1px solid var(--border)" }}
                >
                  <h3
                    className="text-lg font-semibold mb-1"
                    style={{ color: "var(--text-primary)" }}
                  >
                    Confirm Logout
                  </h3>
                  <p
                    className="text-sm mb-5"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Are you sure you want to log out?
                  </p>
                  <div className="flex justify-end gap-2.5">
                    <button
                      onClick={() => setShowLogoutConfirm(false)}
                      className="px-4 py-2 text-sm font-medium rounded-md border hover:bg-[var(--hover-bg)] transition-colors"
                      style={{
                        borderColor: "var(--border)",
                        color: "var(--text-primary)",
                      }}
                    >
                      Cancel
                    </button>
                    <button
                      onClick={confirmLogout}
                      className="px-4 py-2 text-sm font-medium rounded-md text-white transition-colors"
                      style={{ background: "var(--danger)" }}
                    >
                      Logout
                    </button>
                  </div>
                </div>
              </div>,
              portalRootRef.current,
            )}
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
