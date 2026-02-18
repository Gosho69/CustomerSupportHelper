"use client";

import { Mic, BarChart3, TrendingUp, Shield, Zap, Users } from "lucide-react";

const FEATURES = [
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
];

export default function FeaturesSection() {
  return (
    <section id="features" className="py-24" style={{ background: "#ffffff" }}>
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
          {FEATURES.map((feature) => (
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
  );
}
