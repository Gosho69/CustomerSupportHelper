"use client";

import Link from "next/link";
import {
  ArrowRight,
  Mic,
  BarChart3,
  TrendingUp,
  Shield,
  Zap,
  Users,
  Menu,
  X,
  CheckCircle2,
} from "lucide-react";
import { useState } from "react";

export default function LandingPage() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <div className="min-h-screen" style={{ background: "#ffffff" }}>
      {/* Navigation */}
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

            {/* Desktop Navigation + Sign In */}
            <div className="hidden md:flex items-center gap-1">
              {["Features", "How It Works", "Benefits"].map((item) => (
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
              {["Features", "How It Works", "Benefits"].map((item) => (
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

      {/* Hero Section */}
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
            <p
              className="text-sm mt-6"
              style={{ color: "var(--text-tertiary)" }}
            >
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
                  ></div>
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ background: "#ffbd2e" }}
                  ></div>
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ background: "#28c940" }}
                  ></div>
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
                            style={{
                              background: "rgba(255,255,255,0.08)",
                            }}
                          >
                            <div
                              className="h-full rounded-full"
                              style={{
                                background: "var(--accent)",
                                width: `${pct}%`,
                              }}
                            ></div>
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

      {/* Features Section */}
      <section
        id="features"
        className="py-24"
        style={{ background: "#ffffff" }}
      >
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <h2
              className="text-4xl md:text-5xl font-bold mb-4"
              style={{ color: "var(--text-primary)" }}
            >
              Powerful Features for Modern Teams
            </h2>
            <p
              className="text-xl max-w-2xl mx-auto"
              style={{ color: "var(--text-secondary)" }}
            >
              Everything you need to analyze, improve, and scale your customer
              support operations
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                icon: Mic,
                title: "Smart Recording",
                desc: "Automatic call recording with dual-channel support. Capture every conversation with crystal-clear quality and metadata tracking.",
              },
              {
                icon: Zap,
                title: "AI Transcription",
                desc: "Powered by Whisper AI for accurate transcription with speaker diarization, timestamps, and multi-language support.",
              },
              {
                icon: BarChart3,
                title: "Emotion Detection",
                desc: "Advanced sentiment analysis tracks emotions throughout calls — happiness, frustration, anger, and resolution states in real-time.",
              },
              {
                icon: TrendingUp,
                title: "Performance Scoring",
                desc: "Comprehensive rubric evaluates helpfulness, respect, clarity, and policy adherence with explainable, evidence-based ratings.",
              },
              {
                icon: Users,
                title: "AI Coaching",
                desc: "Get personalized coaching tips with specific examples from actual calls. Evidence-based recommendations for continuous improvement.",
              },
              {
                icon: Shield,
                title: "Smart Reports",
                desc: "Automated weekly and monthly reports with actionable insights, trends, and performance metrics for agents and managers.",
              },
            ].map((feature) => (
              <div
                key={feature.title}
                className="rounded-xl p-8 transition-all duration-200 group"
                style={{
                  background: "#ffffff",
                  border: "1px solid var(--border)",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = "var(--accent)";
                  e.currentTarget.style.boxShadow =
                    "0 8px 30px rgba(99,91,255,0.1)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = "var(--border)";
                  e.currentTarget.style.boxShadow = "none";
                }}
              >
                <div
                  className="w-14 h-14 rounded-lg flex items-center justify-center mb-6"
                  style={{ background: "var(--accent-bg)" }}
                >
                  <feature.icon
                    className="w-7 h-7"
                    style={{ color: "var(--accent)" }}
                  />
                </div>
                <h3
                  className="text-2xl font-bold mb-3"
                  style={{ color: "var(--text-primary)" }}
                >
                  {feature.title}
                </h3>
                <p style={{ color: "var(--text-secondary)" }}>{feature.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section
        id="how-it-works"
        className="py-24"
        style={{ background: "var(--background)" }}
      >
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <h2
              className="text-4xl md:text-5xl font-bold mb-4"
              style={{ color: "var(--text-primary)" }}
            >
              How It Works
            </h2>
            <p className="text-xl" style={{ color: "var(--text-secondary)" }}>
              Simple, powerful workflow from recording to insights
            </p>
          </div>

          <div className="max-w-4xl mx-auto space-y-0">
            {[
              {
                step: "1",
                title: "Upload Calls",
                desc: "Upload call recordings or integrate with your phone system. Support for single or dual-channel audio with automatic metadata capture.",
              },
              {
                step: "2",
                title: "AI Analysis",
                desc: "Our AI pipeline transcribes, identifies speakers, detects emotions, analyzes behavior, and scores performance against your quality rubric.",
              },
              {
                step: "3",
                title: "Get Insights",
                desc: "View detailed call analysis with transcripts, emotion timelines, coaching tips, and performance scores — all in one dashboard.",
              },
            ].map((item, i) => (
              <div key={item.step} className="flex items-start gap-8">
                <div className="flex flex-col items-center">
                  <div
                    className="flex-shrink-0 w-14 h-14 rounded-full flex items-center justify-center text-xl font-bold text-white"
                    style={{ background: "var(--accent)" }}
                  >
                    {item.step}
                  </div>
                  {i < 2 && (
                    <div
                      className="w-0.5 h-16 mt-2"
                      style={{ background: "var(--border)" }}
                    ></div>
                  )}
                </div>
                <div className="flex-1 pb-12">
                  <h3
                    className="text-2xl font-bold mb-2"
                    style={{ color: "var(--text-primary)" }}
                  >
                    {item.title}
                  </h3>
                  <p style={{ color: "var(--text-secondary)" }}>{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section
        id="benefits"
        className="py-24"
        style={{ background: "#ffffff" }}
      >
        <div className="container mx-auto px-6">
          <div className="max-w-6xl mx-auto">
            <div className="grid md:grid-cols-2 gap-16 items-center">
              <div>
                <h2
                  className="text-4xl md:text-5xl font-bold mb-8"
                  style={{ color: "var(--text-primary)" }}
                >
                  Measurable Results for Your Team
                </h2>
                <div className="space-y-6">
                  {[
                    {
                      title: "Reduce Handle Time",
                      desc: "Average 23% reduction in call duration through targeted coaching and best practice identification.",
                    },
                    {
                      title: "Improve CSAT",
                      desc: "Customer satisfaction scores increase by 18% on average with evidence-based performance feedback.",
                    },
                    {
                      title: "Scale QA Efficiently",
                      desc: "Analyze 100% of calls automatically instead of random sampling. Catch issues before they escalate.",
                    },
                  ].map((benefit) => (
                    <div key={benefit.title} className="flex items-start gap-4">
                      <div
                        className="flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center mt-0.5"
                        style={{ background: "var(--accent-bg)" }}
                      >
                        <CheckCircle2
                          className="w-5 h-5"
                          style={{ color: "var(--accent)" }}
                        />
                      </div>
                      <div>
                        <h3
                          className="text-xl font-bold mb-1"
                          style={{ color: "var(--text-primary)" }}
                        >
                          {benefit.title}
                        </h3>
                        <p style={{ color: "var(--text-secondary)" }}>
                          {benefit.desc}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="grid grid-cols-1 gap-6">
                {[
                  {
                    label: "Average Score Improvement",
                    value: "+34%",
                    bar: 75,
                  },
                  {
                    label: "Calls Analyzed Monthly",
                    value: "10K+",
                    bar: null,
                  },
                  {
                    label: "Agent Satisfaction",
                    value: "4.8/5",
                    bar: null,
                  },
                ].map((stat) => (
                  <div
                    key={stat.label}
                    className="rounded-xl p-6"
                    style={{
                      background: "#ffffff",
                      border: "1px solid var(--border)",
                    }}
                  >
                    <div
                      className="text-sm mb-2"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {stat.label}
                    </div>
                    <div
                      className="text-4xl font-bold mb-3"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {stat.value}
                    </div>
                    {stat.bar && (
                      <div
                        className="h-2 rounded-full overflow-hidden"
                        style={{ background: "var(--accent-bg)" }}
                      >
                        <div
                          className="h-full rounded-full"
                          style={{
                            background: "var(--accent)",
                            width: `${stat.bar}%`,
                          }}
                        ></div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
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
            {[
              { label: "Features", href: "#features" },
              { label: "How It Works", href: "#how-it-works" },
              { label: "Benefits", href: "#benefits" },
              { label: "Sign In", href: "/login" },
            ].map((link) =>
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
    </div>
  );
}
