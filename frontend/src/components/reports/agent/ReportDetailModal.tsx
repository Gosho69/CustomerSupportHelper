"use client";

import {
  TrendingUp,
  TrendingDown,
  Minus,
  Calendar,
  Award,
  Target,
  Lightbulb,
  Sparkles,
  PhoneCall,
  Clock,
  Users,
  BarChart2,
} from "lucide-react";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  BarChart,
  Bar,
} from "recharts";
import { Report } from "./ReportListItem";

interface ReportDetailModalProps {
  report: Report;
  onClose: () => void;
}

const SENTIMENT_COLORS = {
  Positive: "#0caf60",
  Neutral: "#a3aab8",
  Negative: "#e25950",
};

const SKILL_COLORS = {
  high: "#0caf60",
  mid: "#e68a00",
  low: "#e25950",
};

function skillColor(score: number) {
  if (score >= 75) return SKILL_COLORS.high;
  if (score >= 55) return SKILL_COLORS.mid;
  return SKILL_COLORS.low;
}

function scoreStyle(score: number) {
  if (score >= 85) return { color: "#0caf60" };
  if (score >= 70) return { color: "#e68a00" };
  return { color: "#e25950" };
}

function formatDuration(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = String(seconds % 60).padStart(2, "0");
  return `${m}:${s}`;
}

function RatingBadge({ rating }: { rating?: string }) {
  const map: Record<string, { label: string; bg: string; color: string }> = {
    excellent: { label: "Excellent", bg: "#d1fae5", color: "#065f46" },
    good: { label: "Good", bg: "#fef3c7", color: "#92400e" },
    needs_improvement: {
      label: "Needs Improvement",
      bg: "#fee2e2",
      color: "#991b1b",
    },
    poor: { label: "Poor", bg: "#fee2e2", color: "#991b1b" },
  };
  const style = map[rating ?? ""] ?? {
    label: "No Data",
    bg: "#f3f4f6",
    color: "#6b7280",
  };
  return (
    <span
      className="px-2 py-0.5 rounded-full text-xs font-semibold"
      style={{ background: style.bg, color: style.color }}
    >
      {style.label}
    </span>
  );
}

export default function ReportDetailModal({
  report,
  onClose,
}: ReportDetailModalProps) {
  const sentimentData = [
    { name: "Positive", value: report.positivePercent ?? 0 },
    { name: "Neutral", value: report.neutralPercent ?? 0 },
    { name: "Negative", value: report.negativePercent ?? 0 },
  ].filter((d) => d.value > 0);

  const hasSentiment =
    (report.positivePercent ?? 0) +
      (report.negativePercent ?? 0) +
      (report.neutralPercent ?? 0) >
    0;

  const hasTopics = report.topics && report.topics.length > 0;
  const hasWeeklyTrend =
    report.type === "monthly" &&
    report.weeklyScores &&
    report.weeklyScores.length >= 2;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/30" onClick={onClose} />
      <div
        className="relative rounded-xl w-full max-w-3xl max-h-[92vh] overflow-y-auto shadow-2xl"
        style={{ background: "#ffffff", border: "1px solid var(--border)" }}
      >
        {/* ── Header ─────────────────────────────────────────────── */}
        <div
          className="sticky top-0 z-10 flex items-start justify-between px-8 pt-7 pb-5"
          style={{
            background: "#ffffff",
            borderBottom: "1px solid var(--border)",
          }}
        >
          <div className="flex-1 min-w-0">
            <div className="flex flex-wrap items-center gap-2 mb-1">
              <span
                className="px-3 py-1 rounded-full text-xs font-bold tracking-wide uppercase"
                style={{
                  background: "var(--accent-bg)",
                  color: "var(--accent)",
                }}
              >
                {report.type} Report
              </span>
              <RatingBadge rating={report.rating} />
              {report.aiGenerated && (
                <span
                  className="flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold"
                  style={{ background: "#ede9fe", color: "#5b21b6" }}
                >
                  <Sparkles className="w-3 h-3" />
                  AI Generated
                </span>
              )}
            </div>
            <h2
              className="text-2xl font-bold truncate"
              style={{ color: "var(--text-primary)" }}
            >
              {report.period}
            </h2>
            <p
              className="flex items-center text-sm mt-0.5"
              style={{ color: "var(--text-secondary)" }}
            >
              <Calendar className="w-3.5 h-3.5 mr-1.5" />
              Generated {report.date}
            </p>
          </div>
          <button
            onClick={onClose}
            className="ml-4 w-8 h-8 flex items-center justify-center rounded-lg text-lg font-bold transition-colors hover:bg-gray-100"
            style={{ color: "var(--text-secondary)", flexShrink: 0 }}
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <div className="px-8 py-6 space-y-6">
          {/* ── Executive Summary ──────────────────────────────────── */}
          {report.executiveSummary && (
            <div
              className="rounded-xl p-5"
              style={{
                background: "var(--accent-bg)",
                border: "1px solid var(--border)",
              }}
            >
              <p
                className="text-sm font-semibold mb-1.5 flex items-center gap-1.5"
                style={{ color: "var(--accent)" }}
              >
                <Sparkles className="w-4 h-4" />
                Summary
              </p>
              <p
                className="text-sm leading-relaxed"
                style={{ color: "var(--text-primary)" }}
              >
                {report.executiveSummary}
              </p>
            </div>
          )}

          {/* ── Score + Stats ──────────────────────────────────────── */}
          <div className="grid grid-cols-3 gap-3">
            {/* Overall score */}
            <div
              className="rounded-xl p-4 flex flex-col items-center justify-center text-center"
              style={{
                background: "var(--background)",
                border: "1px solid var(--border)",
              }}
            >
              <p
                className="text-xs mb-1"
                style={{ color: "var(--text-secondary)" }}
              >
                Performance Score
              </p>
              <div className="flex items-center gap-1.5">
                <span
                  className="text-4xl font-bold"
                  style={scoreStyle(report.score)}
                >
                  {report.score}
                </span>
                {report.trend === "up" ? (
                  <TrendingUp
                    className="w-5 h-5"
                    style={{ color: "#0caf60" }}
                  />
                ) : report.trend === "down" ? (
                  <TrendingDown
                    className="w-5 h-5"
                    style={{ color: "#e25950" }}
                  />
                ) : (
                  <Minus
                    className="w-5 h-5"
                    style={{ color: "var(--text-secondary)" }}
                  />
                )}
              </div>
              <p
                className="text-xs mt-0.5 capitalize"
                style={{ color: "var(--text-secondary)" }}
              >
                {report.trend}
              </p>
            </div>

            {/* Total calls */}
            <div
              className="rounded-xl p-4 flex flex-col items-center justify-center text-center"
              style={{
                background: "var(--background)",
                border: "1px solid var(--border)",
              }}
            >
              <PhoneCall
                className="w-5 h-5 mb-1"
                style={{ color: "var(--accent)" }}
              />
              <p
                className="text-xs mb-1"
                style={{ color: "var(--text-secondary)" }}
              >
                Total Calls
              </p>
              <p
                className="text-3xl font-bold"
                style={{ color: "var(--text-primary)" }}
              >
                {report.totalCalls}
              </p>
            </div>

            {/* Avg duration */}
            <div
              className="rounded-xl p-4 flex flex-col items-center justify-center text-center"
              style={{
                background: "var(--background)",
                border: "1px solid var(--border)",
              }}
            >
              <Clock
                className="w-5 h-5 mb-1"
                style={{ color: "var(--accent)" }}
              />
              <p
                className="text-xs mb-1"
                style={{ color: "var(--text-secondary)" }}
              >
                Avg Duration
              </p>
              <p
                className="text-3xl font-bold"
                style={{ color: "var(--text-primary)" }}
              >
                {formatDuration(Math.round(report.avgDuration))}
              </p>
            </div>
          </div>

          {/* ── Quality badges row ─────────────────────────────────── */}
          {(report.consistencyScore != null ||
            report.percentile != null ||
            report.csatScore != null) && (
            <div className="flex flex-wrap gap-3">
              {report.consistencyScore != null && (
                <div
                  className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm"
                  style={{
                    background: "var(--background)",
                    border: "1px solid var(--border)",
                  }}
                >
                  <BarChart2
                    className="w-4 h-4"
                    style={{ color: "var(--accent)" }}
                  />
                  <span style={{ color: "var(--text-secondary)" }}>
                    Consistency
                  </span>
                  <span
                    className="font-bold"
                    style={scoreStyle(Math.round(report.consistencyScore))}
                  >
                    {Math.round(report.consistencyScore)}/100
                  </span>
                </div>
              )}
              {report.percentile != null && (
                <div
                  className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm"
                  style={{
                    background: "var(--background)",
                    border: "1px solid var(--border)",
                  }}
                >
                  <Users
                    className="w-4 h-4"
                    style={{ color: "var(--accent)" }}
                  />
                  <span style={{ color: "var(--text-secondary)" }}>
                    Team Percentile
                  </span>
                  <span
                    className="font-bold"
                    style={{ color: "var(--text-primary)" }}
                  >
                    {Math.round(report.percentile)}th
                  </span>
                </div>
              )}
              {report.csatScore != null && (
                <div
                  className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm"
                  style={{
                    background: "var(--background)",
                    border: "1px solid var(--border)",
                  }}
                >
                  <Award
                    className="w-4 h-4"
                    style={{ color: "var(--accent)" }}
                  />
                  <span style={{ color: "var(--text-secondary)" }}>
                    Avg CSAT
                  </span>
                  <span
                    className="font-bold"
                    style={scoreStyle(((report.csatScore - 1) / 4) * 100)}
                  >
                    {report.csatScore.toFixed(1)}/5
                  </span>
                </div>
              )}
            </div>
          )}

          {/* ── Skills breakdown ───────────────────────────────────── */}
          {report.topSkills && (
            <div
              className="rounded-xl p-5"
              style={{
                background: "var(--background)",
                border: "1px solid var(--border)",
              }}
            >
              <h3
                className="text-sm font-semibold mb-4 flex items-center gap-2"
                style={{ color: "var(--text-primary)" }}
              >
                <Target
                  className="w-4 h-4"
                  style={{ color: "var(--accent)" }}
                />
                Skills Breakdown
              </h3>
              <div className="space-y-3">
                {report.topSkills.map(({ skill, score }) => (
                  <div key={skill}>
                    <div className="flex justify-between items-center mb-1">
                      <span
                        className="text-sm"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        {skill}
                      </span>
                      <span
                        className="text-sm font-bold"
                        style={{ color: skillColor(score) }}
                      >
                        {score}%
                      </span>
                    </div>
                    <div
                      className="w-full rounded-full h-2"
                      style={{ background: "#e5e7eb" }}
                    >
                      <div
                        className="h-2 rounded-full transition-all"
                        style={{
                          width: `${score}%`,
                          background: skillColor(score),
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── Sentiment + Topics (side by side) ─────────────────── */}
          {(hasSentiment || hasTopics) && (
            <div
              className={`grid gap-4 ${hasSentiment && hasTopics ? "grid-cols-2" : "grid-cols-1"}`}
            >
              {/* Sentiment donut */}
              {hasSentiment && (
                <div
                  className="rounded-xl p-5"
                  style={{
                    background: "var(--background)",
                    border: "1px solid var(--border)",
                  }}
                >
                  <h3
                    className="text-sm font-semibold mb-3 flex items-center gap-2"
                    style={{ color: "var(--text-primary)" }}
                  >
                    <PhoneCall
                      className="w-4 h-4"
                      style={{ color: "var(--accent)" }}
                    />
                    Call Sentiment
                  </h3>
                  <ResponsiveContainer width="100%" height={150}>
                    <PieChart>
                      <Pie
                        data={sentimentData}
                        cx="50%"
                        cy="50%"
                        innerRadius={42}
                        outerRadius={62}
                        paddingAngle={3}
                        dataKey="value"
                      >
                        {sentimentData.map((entry) => (
                          <Cell
                            key={entry.name}
                            fill={
                              SENTIMENT_COLORS[
                                entry.name as keyof typeof SENTIMENT_COLORS
                              ]
                            }
                          />
                        ))}
                      </Pie>
                      <Tooltip
                        formatter={(v: number | undefined) => [
                          v != null ? `${v}%` : "",
                          "",
                        ]}
                        contentStyle={{
                          background: "#fff",
                          border: "1px solid #e3e8ee",
                          borderRadius: "8px",
                          fontSize: "12px",
                        }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="flex justify-center gap-4 mt-1">
                    {sentimentData.map((d) => (
                      <div
                        key={d.name}
                        className="flex items-center gap-1.5 text-xs"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        <span
                          className="w-2.5 h-2.5 rounded-full"
                          style={{
                            background:
                              SENTIMENT_COLORS[
                                d.name as keyof typeof SENTIMENT_COLORS
                              ],
                          }}
                        />
                        {d.name} {d.value}%
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Topics bar chart */}
              {hasTopics && (
                <div
                  className="rounded-xl p-5"
                  style={{
                    background: "var(--background)",
                    border: "1px solid var(--border)",
                  }}
                >
                  <h3
                    className="text-sm font-semibold mb-3 flex items-center gap-2"
                    style={{ color: "var(--text-primary)" }}
                  >
                    <BarChart2
                      className="w-4 h-4"
                      style={{ color: "var(--accent)" }}
                    />
                    Top Call Topics
                  </h3>
                  <ResponsiveContainer width="100%" height={160}>
                    <BarChart
                      data={report.topics}
                      layout="vertical"
                      margin={{ left: 0, right: 16, top: 0, bottom: 0 }}
                    >
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="#e5e7eb"
                        horizontal={false}
                      />
                      <XAxis type="number" tick={{ fontSize: 11 }} />
                      <YAxis
                        dataKey="topic"
                        type="category"
                        width={90}
                        tick={{ fontSize: 11 }}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#fff",
                          border: "1px solid #e3e8ee",
                          borderRadius: "8px",
                          fontSize: "12px",
                        }}
                      />
                      <Bar
                        dataKey="count"
                        fill="var(--accent, #635bff)"
                        radius={[0, 4, 4, 0]}
                        name="Calls"
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          )}

          {/* ── Monthly: Weekly Trend Chart ────────────────────────── */}
          {hasWeeklyTrend && (
            <div
              className="rounded-xl p-5"
              style={{
                background: "var(--background)",
                border: "1px solid var(--border)",
              }}
            >
              <h3
                className="text-sm font-semibold mb-4 flex items-center gap-2"
                style={{ color: "var(--text-primary)" }}
              >
                <TrendingUp
                  className="w-4 h-4"
                  style={{ color: "var(--accent)" }}
                />
                Weekly Performance Trend
              </h3>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart
                  data={report.weeklyScores}
                  margin={{ left: 0, right: 12, top: 4, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="label" tick={{ fontSize: 12 }} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{
                      background: "#fff",
                      border: "1px solid #e3e8ee",
                      borderRadius: "8px",
                      fontSize: "12px",
                    }}
                    formatter={(v: number | undefined) => [
                      v != null ? `${v}` : "",
                      "Score",
                    ]}
                  />
                  <Line
                    type="monotone"
                    dataKey="score"
                    stroke="var(--accent, #635bff)"
                    strokeWidth={2.5}
                    dot={{ r: 4, fill: "var(--accent, #635bff)" }}
                    activeDot={{ r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* ── Strengths + Improvements ───────────────────────────── */}
          <div className="grid grid-cols-2 gap-4">
            <div
              className="rounded-xl p-5"
              style={{
                background: "var(--background)",
                border: "1px solid var(--border)",
              }}
            >
              <h3
                className="text-sm font-semibold mb-3 flex items-center gap-2"
                style={{ color: "var(--text-primary)" }}
              >
                <Award className="w-4 h-4" style={{ color: "#0caf60" }} />
                Strengths
              </h3>
              {report.strengths.length > 0 ? (
                <ul className="space-y-2">
                  {report.strengths.map((s, i) => (
                    <li
                      key={i}
                      className="flex items-start gap-2 text-sm"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      <span
                        className="mt-0.5 font-bold"
                        style={{ color: "#0caf60" }}
                      >
                        ✓
                      </span>
                      {s}
                    </li>
                  ))}
                </ul>
              ) : (
                <p
                  className="text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
                  No strengths identified yet.
                </p>
              )}
            </div>

            <div
              className="rounded-xl p-5"
              style={{
                background: "var(--background)",
                border: "1px solid var(--border)",
              }}
            >
              <h3
                className="text-sm font-semibold mb-3 flex items-center gap-2"
                style={{ color: "var(--text-primary)" }}
              >
                <Target className="w-4 h-4" style={{ color: "#e68a00" }} />
                Areas for Improvement
              </h3>
              {report.improvements.length > 0 ? (
                <ul className="space-y-2">
                  {report.improvements.map((item, i) => (
                    <li
                      key={i}
                      className="flex items-start gap-2 text-sm"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      <span
                        className="mt-0.5 font-bold"
                        style={{ color: "#e68a00" }}
                      >
                        →
                      </span>
                      {item}
                    </li>
                  ))}
                </ul>
              ) : (
                <p
                  className="text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
                  No improvement areas identified.
                </p>
              )}
            </div>
          </div>

          {/* ── Recommendations ────────────────────────────────────── */}
          {report.recommendations && report.recommendations.length > 0 && (
            <div
              className="rounded-xl p-5"
              style={{
                background: "#fefce8",
                border: "1px solid #fde68a",
              }}
            >
              <h3
                className="text-sm font-semibold mb-3 flex items-center gap-2"
                style={{ color: "#92400e" }}
              >
                <Lightbulb className="w-4 h-4" />
                Coaching Recommendations
              </h3>
              <ul className="space-y-2">
                {report.recommendations.map((rec, i) => (
                  <li
                    key={i}
                    className="flex items-start gap-2 text-sm"
                    style={{ color: "#78350f" }}
                  >
                    <span className="mt-0.5 font-bold">•</span>
                    {rec}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* ── Footer ─────────────────────────────────────────────── */}
        <div
          className="sticky bottom-0 px-8 py-4 flex justify-end"
          style={{
            background: "#ffffff",
            borderTop: "1px solid var(--border)",
          }}
        >
          <button
            onClick={onClose}
            className="px-6 py-2.5 rounded-lg font-semibold text-sm transition-colors"
            style={{
              background: "var(--accent-bg)",
              color: "var(--text-primary)",
            }}
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
