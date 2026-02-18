"use client";

import Link from "next/link";
import { ArrowRight, Zap } from "lucide-react";

export default function HeroSection() {
  return (
    <section
      className="pt-32 pb-20"
      style={{
        background: "linear-gradient(180deg, #f0efff 0%, #ffffff 100%)",
      }}
    >
      <div className="container mx-auto px-6 text-center">
        <div className="max-w-4xl mx-auto">
          <div
            className="inline-flex items-center px-4 py-1.5 rounded-full text-sm font-medium mb-8"
            style={{
              background: "var(--accent-bg)",
              color: "var(--accent)",
              border: "1px solid rgba(99,91,255,0.2)",
            }}
          >
            <Zap className="w-4 h-4 mr-2" />
            AI-Powered Call Analytics
          </div>
          <h1
            className="text-5xl md:text-7xl font-bold mb-6 leading-tight"
            style={{ color: "var(--text-primary)" }}
          >
            Transform Customer{" "}
            <span style={{ color: "var(--accent)" }}>Support</span> with
            Intelligent Insights
          </h1>
          <p
            className="text-xl mb-10 leading-relaxed max-w-2xl mx-auto"
            style={{ color: "var(--text-secondary)" }}
          >
            Record, transcribe, and analyze every customer interaction. Get
            actionable insights, emotion detection, and AI-driven coaching to
            elevate your support team&apos;s performance.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/login"
              className="px-8 py-4 rounded-lg transition font-semibold text-lg flex items-center justify-center group text-white"
              style={{ background: "var(--accent)" }}
            >
              Start Free Trial
              <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition" />
            </Link>
            <a
              href="#how-it-works"
              className="px-8 py-4 rounded-lg transition font-semibold text-lg"
              style={{
                color: "var(--text-primary)",
                border: "1px solid var(--border)",
                background: "#ffffff",
              }}
            >
              Learn More
            </a>
          </div>
          <p className="text-sm mt-6" style={{ color: "var(--text-tertiary)" }}>
            No credit card required • 14-day free trial
          </p>
        </div>

        {/* Hero Dashboard Preview */}
        <div className="mt-16 max-w-5xl mx-auto">
          <div
            className="rounded-2xl p-1"
            style={{
              background:
                "linear-gradient(135deg, var(--accent), var(--accent-light), #a78bfa)",
              boxShadow:
                "0 25px 60px rgba(99,91,255,0.25), 0 10px 20px rgba(0,0,0,0.08)",
            }}
          >
            <div
              className="rounded-xl overflow-hidden"
              style={{ background: "#1a1f36" }}
            >
              <div
                className="flex items-center space-x-2 px-4 py-3"
                style={{ borderBottom: "1px solid rgba(255,255,255,0.08)" }}
              >
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ background: "#ff5f57" }}
                />
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ background: "#ffbd2e" }}
                />
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ background: "#28c940" }}
                />
              </div>
              <div className="p-8 space-y-4">
                <div className="grid grid-cols-3 gap-4">
                  {[
                    { label: "Avg. Score", value: "8.5/10" },
                    { label: "Calls Analyzed", value: "1,247" },
                    { label: "Sentiment", value: "+12%" },
                  ].map((stat) => (
                    <div
                      key={stat.label}
                      className="rounded-lg p-4"
                      style={{
                        background: "rgba(255,255,255,0.06)",
                        border: "1px solid rgba(255,255,255,0.08)",
                      }}
                    >
                      <div
                        className="text-sm mb-1"
                        style={{ color: "rgba(255,255,255,0.6)" }}
                      >
                        {stat.label}
                      </div>
                      <div className="text-3xl font-bold text-white">
                        {stat.value}
                      </div>
                    </div>
                  ))}
                </div>
                <div
                  className="rounded-lg p-6"
                  style={{
                    background: "rgba(255,255,255,0.04)",
                    border: "1px solid rgba(255,255,255,0.06)",
                  }}
                >
                  <div className="flex items-center justify-between mb-4">
                    <span style={{ color: "rgba(255,255,255,0.5)" }}>
                      Recent Call Analysis
                    </span>
                    <span
                      className="text-sm px-2 py-0.5 rounded-full"
                      style={{
                        color: "#28c940",
                        background: "rgba(40,201,64,0.15)",
                      }}
                    >
                      Live
                    </span>
                  </div>
                  <div className="space-y-3">
                    {[85, 92, 78].map((pct, i) => (
                      <div key={i} className="flex items-center gap-3">
                        <span
                          className="text-xs w-16"
                          style={{ color: "rgba(255,255,255,0.4)" }}
                        >
                          Call #{i + 1}
                        </span>
                        <div
                          className="flex-1 h-2 rounded-full overflow-hidden"
                          style={{ background: "rgba(255,255,255,0.08)" }}
                        >
                          <div
                            className="h-full rounded-full"
                            style={{
                              background: "var(--accent)",
                              width: `${pct}%`,
                            }}
                          />
                        </div>
                        <span
                          className="text-xs w-10 text-right"
                          style={{ color: "rgba(255,255,255,0.5)" }}
                        >
                          {pct}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
