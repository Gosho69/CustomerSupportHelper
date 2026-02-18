"use client";

import {
  Smile,
  Heart,
  Activity,
  Award,
  Clock,
  MessageSquare,
  CheckCircle,
} from "lucide-react";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import {
  CallDetail,
  EMOTION_COLORS,
  formatDuration,
  getSentimentColor,
  getScoreColor,
} from "./callDetailHelpers";

interface OverviewTabProps {
  call: CallDetail;
}

export default function OverviewTab({ call }: OverviewTabProps) {
  // Backend: emotion_distribution = { emotion: { count, percentage } }
  const rawDist = call.emotional_summary?.emotion_distribution;
  const emotionData = rawDist
    ? Object.entries(rawDist).map(([name, val]: [string, any]) => ({
        name: name.charAt(0).toUpperCase() + name.slice(1),
        value: typeof val === "object" ? val.count || val.percentage || 0 : val,
      }))
    : [];

  // Backend: behavioral_analysis fields
  const ba = call.behavioral_analysis || {};
  const behaviorRadarData = [
    {
      metric: "Response Time",
      score:
        (ba.response_time_analysis?.avg_agent_response_time ?? 0) <= 1.5
          ? 95
          : (ba.response_time_analysis?.avg_agent_response_time ?? 0) <= 3
            ? 70
            : 50,
    },
    {
      metric: "Speaking Pace",
      score:
        ba.words_per_minute?.agent_pacing === "optimal"
          ? 90
          : ba.words_per_minute?.agent_pacing === "slightly_fast"
            ? 75
            : 60,
    },
    {
      metric: "Active Listening",
      score:
        ba.active_listening?.listening_assessment === "excellent"
          ? 95
          : ba.active_listening?.listening_assessment === "good"
            ? 75
            : 50,
    },
    {
      metric: "Interruptions",
      score: Math.max(
        0,
        100 - (ba.interruption_analysis?.agent_interruptions || 0) * 15,
      ),
    },
    {
      metric: "Questions",
      score: Math.min(100, (ba.question_analysis?.agent_questions || 0) * 20),
    },
  ];

  const summaryText =
    call.transcript_summary?.summary ||
    (typeof call.transcript_summary === "string"
      ? call.transcript_summary
      : null) ||
    "No summary available for this call.";

  const overallSentiment =
    call.emotional_summary?.call_tone ||
    call.emotional_summary?.overall_sentiment ||
    "neutral";

  return (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div
          className="rounded-lg p-4"
          style={{
            background: "var(--background)",
            border: "1px solid var(--border)",
          }}
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                Overall Score
              </p>
              <p
                className={`text-3xl font-bold mt-1 ${getScoreColor(
                  ba.behavioral_score || 0,
                )}`}
              >
                {Math.round(ba.behavioral_score || 0)}
                <span className="text-lg">/100</span>
              </p>
            </div>
            <Award className="w-10 h-10" style={{ color: "#e68a00" }} />
          </div>
        </div>

        <div
          className="rounded-lg p-4"
          style={{
            background: "var(--background)",
            border: "1px solid var(--border)",
          }}
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                Sentiment
              </p>
              <p
                className={`text-xl font-bold mt-1 capitalize ${getSentimentColor(
                  overallSentiment,
                )}`}
              >
                {overallSentiment.charAt(0).toUpperCase() +
                  overallSentiment.slice(1)}
              </p>
            </div>
            <Heart className="w-10 h-10" style={{ color: "#ec4899" }} />
          </div>
        </div>

        <div
          className="rounded-lg p-4"
          style={{
            background: "var(--background)",
            border: "1px solid var(--border)",
          }}
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                Duration
              </p>
              <p
                className="text-3xl font-bold mt-1"
                style={{ color: "var(--text-primary)" }}
              >
                {formatDuration(call.duration)}
              </p>
            </div>
            <Clock className="w-10 h-10" style={{ color: "var(--accent)" }} />
          </div>
        </div>
      </div>

      {/* GPT-4 Ratings */}
      {call.transcript_summary?.detailed_ratings && (
        <div
          className="rounded-lg p-6"
          style={{
            background: "var(--background)",
            border: "1px solid var(--border)",
          }}
        >
          <h3
            className="text-lg font-semibold mb-4 flex items-center"
            style={{ color: "var(--text-primary)" }}
          >
            <Award className="w-5 h-5 mr-2" style={{ color: "#e68a00" }} />
            Performance Ratings
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {Object.entries(call.transcript_summary.detailed_ratings).map(
              ([key, value]: [string, any]) => (
                <div key={key} className="text-center">
                  <p
                    className="text-sm capitalize mb-1"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {key}
                  </p>
                  <p
                    className={`text-2xl font-bold ${
                      value >= 4
                        ? "text-green-600"
                        : value >= 3
                          ? "text-yellow-600"
                          : "text-red-600"
                    }`}
                  >
                    {value}
                    <span className="text-sm text-gray-400">/5</span>
                  </p>
                </div>
              ),
            )}
          </div>
        </div>
      )}

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Emotion Distribution Pie Chart */}
        <div
          className="rounded-lg p-6"
          style={{
            background: "var(--background)",
            border: "1px solid var(--border)",
          }}
        >
          <h3
            className="text-lg font-semibold mb-4 flex items-center"
            style={{ color: "var(--text-primary)" }}
          >
            <Smile className="w-5 h-5 mr-2" style={{ color: "#e68a00" }} />
            Emotion Distribution
          </h3>
          {emotionData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={emotionData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) =>
                    `${name}: ${((percent || 0) * 100).toFixed(0)}%`
                  }
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {emotionData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={
                        EMOTION_COLORS[entry.name.toLowerCase()] || "#6b7280"
                      }
                    />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#ffffff",
                    border: "1px solid var(--border)",
                    borderRadius: "0.5rem",
                    color: "var(--text-primary)",
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <p
              className="text-center py-12"
              style={{ color: "var(--text-secondary)" }}
            >
              No emotion data available
            </p>
          )}
        </div>

        {/* Behavioral Radar Chart */}
        <div
          className="rounded-lg p-6"
          style={{
            background: "var(--background)",
            border: "1px solid var(--border)",
          }}
        >
          <h3
            className="text-lg font-semibold mb-4 flex items-center"
            style={{ color: "var(--text-primary)" }}
          >
            <Activity
              className="w-5 h-5 mr-2"
              style={{ color: "var(--accent)" }}
            />
            Performance Radar
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={behaviorRadarData}>
              <PolarGrid stroke="#e3e8ee" />
              <PolarAngleAxis
                dataKey="metric"
                tick={{ fill: "#697386", fontSize: 12 }}
              />
              <PolarRadiusAxis
                angle={90}
                domain={[0, 100]}
                tick={{ fill: "#697386" }}
              />
              <Radar
                name="Performance"
                dataKey="score"
                stroke="#06b6d4"
                fill="#06b6d4"
                fillOpacity={0.6}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#ffffff",
                  border: "1px solid var(--border)",
                  borderRadius: "0.5rem",
                  color: "var(--text-primary)",
                }}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Summary Section */}
      <div
        className="rounded-lg p-6"
        style={{
          background: "var(--background)",
          border: "1px solid var(--border)",
        }}
      >
        <h3
          className="text-lg font-semibold mb-4 flex items-center"
          style={{ color: "var(--text-primary)" }}
        >
          <MessageSquare
            className="w-5 h-5 mr-2"
            style={{ color: "var(--accent)" }}
          />
          Call Summary
        </h3>
        <p
          className="leading-relaxed"
          style={{ color: "var(--text-secondary)" }}
        >
          {summaryText}
        </p>

        {/* Tone info from GPT-4 */}
        {(call.transcript_summary?.customer_tone ||
          call.transcript_summary?.agent_tone) && (
          <div className="mt-4 flex items-center space-x-6">
            {call.transcript_summary.customer_tone && (
              <div>
                <span
                  className="text-sm font-medium"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Customer Tone:{" "}
                </span>
                <span
                  className="text-sm font-semibold capitalize"
                  style={{ color: "var(--text-primary)" }}
                >
                  {call.transcript_summary.customer_tone}
                </span>
              </div>
            )}
            {call.transcript_summary.agent_tone && (
              <div>
                <span
                  className="text-sm font-medium"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Agent Tone:{" "}
                </span>
                <span
                  className="text-sm font-semibold capitalize"
                  style={{ color: "var(--text-primary)" }}
                >
                  {call.transcript_summary.agent_tone}
                </span>
              </div>
            )}
          </div>
        )}

        {call.transcript_summary?.key_points &&
          call.transcript_summary.key_points.length > 0 && (
            <div className="mt-4">
              <p
                className="text-sm font-medium mb-2"
                style={{ color: "var(--text-secondary)" }}
              >
                Key Points:
              </p>
              <ul className="space-y-2">
                {call.transcript_summary.key_points.map(
                  (point: string, index: number) => (
                    <li key={index} className="flex items-start">
                      <CheckCircle
                        className="w-4 h-4 mr-2 mt-0.5 flex-shrink-0"
                        style={{ color: "#0caf60" }}
                      />
                      <span style={{ color: "var(--text-secondary)" }}>
                        {point}
                      </span>
                    </li>
                  ),
                )}
              </ul>
            </div>
          )}
      </div>

      {/* Emotional Summary Narrative */}
      {call.emotional_summary?.summary && (
        <div
          className="rounded-lg p-6"
          style={{
            background: "var(--background)",
            border: "1px solid var(--border)",
          }}
        >
          <h3
            className="text-lg font-semibold mb-4 flex items-center"
            style={{ color: "var(--text-primary)" }}
          >
            <Heart className="w-5 h-5 mr-2" style={{ color: "#ec4899" }} />
            Emotional Summary
          </h3>
          <p
            className="leading-relaxed"
            style={{ color: "var(--text-secondary)" }}
          >
            {call.emotional_summary.summary}
          </p>
          <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
            {call.emotional_summary.resolution_status && (
              <div>
                <p
                  className="text-xs"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Resolution
                </p>
                <p
                  className="font-semibold capitalize"
                  style={{ color: "var(--text-primary)" }}
                >
                  {call.emotional_summary.resolution_status.replace(/_/g, " ")}
                </p>
              </div>
            )}
            {call.emotional_summary.customer_satisfaction && (
              <div>
                <p
                  className="text-xs"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Satisfaction
                </p>
                <p
                  className="font-semibold capitalize"
                  style={{ color: "var(--text-primary)" }}
                >
                  {call.emotional_summary.customer_satisfaction.replace(
                    /_/g,
                    " ",
                  )}
                </p>
              </div>
            )}
            {call.emotional_summary.agent_empathy_score !== undefined && (
              <div>
                <p
                  className="text-xs"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Agent Empathy
                </p>
                <p
                  className="font-semibold"
                  style={{ color: "var(--text-primary)" }}
                >
                  {(call.emotional_summary.agent_empathy_score * 100).toFixed(
                    0,
                  )}
                  %
                </p>
              </div>
            )}
            {call.emotional_summary.customer_frustration_level !==
              undefined && (
              <div>
                <p
                  className="text-xs"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Frustration Level
                </p>
                <p
                  className="font-semibold"
                  style={{ color: "var(--text-primary)" }}
                >
                  {(
                    call.emotional_summary.customer_frustration_level * 100
                  ).toFixed(0)}
                  %
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
