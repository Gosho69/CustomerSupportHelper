"use client";

import Link from "next/link";
import { Mic, Menu, X } from "lucide-react";
import { useState } from "react";

const NAV_ITEMS = ["Features", "How It Works", "Benefits"];

export default function Navbar() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <nav
      className="fixed top-0 left-0 right-0 z-50 backdrop-blur-md"
      style={{
        background: "rgba(255,255,255,0.92)",
        borderBottom: "1px solid var(--border)",
      }}
    >
      <div className="max-w-6xl mx-auto px-6 py-3">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <Link href="/" className="flex items-center space-x-2 group">
            <div
              className="w-9 h-9 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform duration-200"
              style={{ background: "var(--accent)" }}
            >
              <Mic className="w-5 h-5 text-white" />
            </div>
            <span
              className="text-xl font-bold"
              style={{ color: "var(--text-primary)" }}
            >
              AgentSights
            </span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-1">
            {NAV_ITEMS.map((item) => (
              <a
                key={item}
                href={`#${item.toLowerCase().replace(/ /g, "-")}`}
                className="px-3.5 py-2 rounded-lg text-sm font-medium transition-all duration-200"
                style={{ color: "var(--text-secondary)" }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.color = "var(--text-primary)";
                  e.currentTarget.style.background = "var(--hover-bg)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.color = "var(--text-secondary)";
                  e.currentTarget.style.background = "transparent";
                }}
              >
                {item}
              </a>
            ))}
            <div
              className="w-px h-5 mx-2"
              style={{ background: "var(--border)" }}
            />
            <Link
              href="/login"
              className="px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 text-white"
              style={{ background: "var(--accent)" }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "var(--accent-light)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "var(--accent)";
              }}
            >
              Sign In
            </Link>
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 rounded-lg transition-all duration-200"
            style={{ color: "var(--text-secondary)" }}
            aria-label="Toggle menu"
          >
            {mobileMenuOpen ? (
              <X className="w-6 h-6" />
            ) : (
              <Menu className="w-6 h-6" />
            )}
          </button>
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <div
            className="md:hidden mt-4 pb-4 space-y-2 pt-4"
            style={{ borderTop: "1px solid var(--border)" }}
          >
            {NAV_ITEMS.map((item) => (
              <a
                key={item}
                href={`#${item.toLowerCase().replace(/ /g, "-")}`}
                onClick={() => setMobileMenuOpen(false)}
                className="block px-4 py-3 rounded-lg transition-all duration-200"
                style={{ color: "var(--text-secondary)" }}
              >
                {item}
              </a>
            ))}
            <div
              className="pt-4 space-y-2 mt-4"
              style={{ borderTop: "1px solid var(--border)" }}
            >
              <Link
                href="/login"
                onClick={() => setMobileMenuOpen(false)}
                className="block px-4 py-3 text-center rounded-lg font-medium text-white"
                style={{ background: "var(--accent)" }}
              >
                Sign In
              </Link>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}
