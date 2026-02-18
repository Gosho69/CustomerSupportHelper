"use client";

import Link from "next/link";
import { Mic } from "lucide-react";

const LINKS = [
  { label: "Features", href: "#features" },
  { label: "How It Works", href: "#how-it-works" },
  { label: "Benefits", href: "#benefits" },
  { label: "Sign In", href: "/login" },
];

export default function Footer() {
  return (
    <footer
      className="py-8"
      style={{
        background: "#ffffff",
        borderTop: "1px solid var(--border)",
      }}
    >
      <div className="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center space-x-2">
          <div
            className="w-7 h-7 rounded-md flex items-center justify-center"
            style={{ background: "var(--accent)" }}
          >
            <Mic className="w-4 h-4 text-white" />
          </div>
          <span
            className="text-sm font-semibold"
            style={{ color: "var(--text-primary)" }}
          >
            AgentSights
          </span>
        </div>
        <div className="flex items-center gap-6 text-sm">
          {LINKS.map((link) =>
            link.href.startsWith("/") ? (
              <Link
                key={link.label}
                href={link.href}
                className="transition-colors"
                style={{ color: "var(--text-secondary)" }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.color = "var(--accent)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.color = "var(--text-secondary)";
                }}
              >
                {link.label}
              </Link>
            ) : (
              <a
                key={link.label}
                href={link.href}
                className="transition-colors"
                style={{ color: "var(--text-secondary)" }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.color = "var(--accent)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.color = "var(--text-secondary)";
                }}
              >
                {link.label}
              </a>
            ),
          )}
        </div>
        <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
          &copy; {new Date().getFullYear()} AgentSights
        </p>
      </div>
    </footer>
  );
}
