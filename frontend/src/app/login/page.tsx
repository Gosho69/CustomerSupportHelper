"use client";

import Link from "next/link";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Eye, EyeOff, Loader2, AlertCircle } from "lucide-react";
import { useAuthStore } from "@/store/authStore";
import { authApi } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const { setAuth, isAuthenticated } = useAuthStore();
  const [formData, setFormData] = useState({
    username: "",
    password: "",
  });
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated()) {
      router.push("/dashboard");
    }
  }, [isAuthenticated, router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      const response = await authApi.login(
        formData.username,
        formData.password,
      );
      const data = response.data;

      // Store user in Zustand — tokens are in httpOnly cookies set by the server
      setAuth(data.user);

      router.push("/dashboard");
    } catch (err: any) {
      const errorMessage =
        err.response?.data?.error ||
        err.message ||
        "Invalid credentials. Please try again.";
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div
      className="min-h-screen flex items-center justify-center p-6"
      style={{ background: "var(--background)" }}
    >
      {/* Login Card */}
      <div className="relative w-full max-w-md">
        {/* Logo and Brand */}
        <div className="text-center mb-8">
          <Link href="/" className="inline-flex items-center space-x-3 group">
            <img
              src="/logo-icon.svg"
              alt="AgentSights"
              className="w-14 h-14 group-hover:scale-105 transition-transform duration-200"
            />
          </Link>
          <h1
            className="text-2xl font-bold mt-5"
            style={{ color: "var(--text-primary)" }}
          >
            Sign in to AgentSights
          </h1>
          <p
            className="mt-2 text-sm"
            style={{ color: "var(--text-secondary)" }}
          >
            Welcome back! Enter your credentials to continue
          </p>
        </div>

        {/* Login Form */}
        <div
          className="rounded-2xl p-8"
          style={{
            background: "#ffffff",
            border: "1px solid var(--border)",
            boxShadow: "0 4px 24px rgba(0,0,0,0.06)",
          }}
        >
          {error && (
            <div
              className="mb-6 p-4 rounded-xl flex gap-3"
              style={{
                background: "var(--danger-bg)",
                border: "1px solid rgba(223,27,65,0.15)",
              }}
            >
              <AlertCircle
                className="w-5 h-5 flex-shrink-0 mt-0.5"
                style={{ color: "var(--danger)" }}
              />
              <p className="text-sm" style={{ color: "var(--danger)" }}>
                {error}
              </p>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Username Field */}
            <div>
              <label
                htmlFor="username"
                className="block text-sm font-medium mb-2"
                style={{ color: "var(--text-primary)" }}
              >
                Username
              </label>
              <input
                id="username"
                type="text"
                required
                value={formData.username}
                onChange={(e) =>
                  setFormData({ ...formData, username: e.target.value })
                }
                className="w-full px-4 py-3 rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
                style={{
                  background: "#ffffff",
                  border: "1px solid var(--input-border)",
                  color: "var(--text-primary)",
                }}
                placeholder="Enter your username"
                disabled={isLoading}
              />
            </div>

            {/* Password Field */}
            <div>
              <label
                htmlFor="password"
                className="block text-sm font-medium mb-2"
                style={{ color: "var(--text-primary)" }}
              >
                Password
              </label>
              <div className="relative">
                <input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  required
                  value={formData.password}
                  onChange={(e) =>
                    setFormData({ ...formData, password: e.target.value })
                  }
                  className="w-full px-4 py-3 rounded-lg transition-all duration-200 pr-12 focus:outline-none focus:ring-2"
                  style={{
                    background: "#ffffff",
                    border: "1px solid var(--input-border)",
                    color: "var(--text-primary)",
                  }}
                  placeholder="Enter your password"
                  disabled={isLoading}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 transition-colors"
                  style={{ color: "var(--text-tertiary)" }}
                  disabled={isLoading}
                  aria-label="toggle password visibility"
                >
                  {showPassword ? (
                    <EyeOff className="w-5 h-5" />
                  ) : (
                    <Eye className="w-5 h-5" />
                  )}
                </button>
              </div>
            </div>

            {/* Forgot Password */}
            <div className="flex justify-end text-sm">
              <a
                href="#"
                className="transition-colors"
                style={{ color: "var(--accent)" }}
              >
                Forgot password?
              </a>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full py-3.5 rounded-lg font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2 text-white"
              style={{
                background: "var(--accent)",
              }}
              onMouseEnter={(e) => {
                if (!isLoading)
                  e.currentTarget.style.background = "var(--accent-light)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "var(--accent)";
              }}
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Signing in...</span>
                </>
              ) : (
                <span>Sign In</span>
              )}
            </button>
          </form>
        </div>

        {/* Footer */}
        <p
          className="text-center text-sm mt-8"
          style={{ color: "var(--text-tertiary)" }}
        >
          Don&apos;t have an account?{" "}
          <Link
            href="/"
            className="font-medium transition-colors"
            style={{ color: "var(--accent)" }}
          >
            Learn more
          </Link>
        </p>
      </div>
    </div>
  );
}
