"use client";

const STEPS = [
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
];

export default function HowItWorksSection() {
  return (
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
          {STEPS.map((item, i) => (
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
                  />
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
  );
}
