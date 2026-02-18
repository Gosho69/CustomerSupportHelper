"use client";

import { CheckCircle2 } from "lucide-react";

const BENEFITS = [
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
];

const STATS = [
  { label: "Average Score Improvement", value: "+34%", bar: 75 },
  { label: "Calls Analyzed Monthly", value: "10K+", bar: null },
  { label: "Agent Satisfaction", value: "4.8/5", bar: null },
];

export default function BenefitsSection() {
  return (
    <section id="benefits" className="py-24" style={{ background: "#ffffff" }}>
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
                {BENEFITS.map((benefit) => (
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
              {STATS.map((stat) => (
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
                      />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
