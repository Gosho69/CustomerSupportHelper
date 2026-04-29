"use client";

import {
  Heart,
  Clock,
  AlertCircle,
  MessageSquare,
  Activity,
  BarChart3,
  TrendingUp,
  TrendingDown,
  CheckCircle,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { CallDetail } from "./callDetailHelpers";

interface BehaviorTabProps {
  call: CallDetail;
}

export default function BehaviorTab({ call }: BehaviorTabProps) {
  const ba = call.behavioral_analysis || {};
  const bs = call.behavioral_summary || {};

  const wpmData = [
    { name: "Agent",    wpm: ba.words_per_minute?.agent_wpm    || 0 },
    { name: "Customer", wpm: ba.words_per_minute?.customer_wpm || 0 },
  ];

  // Backend: key_strengths = [{ type, description }], NOT strengths (plain strings)
  const strengths = bs.key_strengths || [];
  // Backend: key_issues = [{ type, severity, description }], NOT areas_for_improvement
  const issues = bs.key_issues || [];

  return (
    <div className="space-y-6">
      {/* Words Per Minute Chart */}
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
          <BarChart3
            className="w-5 h-5 mr-2"
            style={{ color: "var(--accent)" }}
          />
          Speaking Pace (Words Per Minute)
        </h3>
        <p className="text-xs mb-3" style={{ color: "var(--text-tertiary)" }}>
          Green lines mark the optimal range (120–180 wpm). Above 180 = slightly fast; above 220 = too fast.
        </p>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={wpmData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ee" />
            <XAxis dataKey="name" tick={{ fill: "#697386" }} />
            <YAxis tick={{ fill: "#697386" }} domain={[0, 260]} />
            <Tooltip
              contentStyle={{
                backgroundColor: "#ffffff",
                border: "1px solid var(--border)",
                borderRadius: "0.5rem",
                color: "var(--text-primary)",
              }}
            />
            <Legend />
            <ReferenceLine y={180} stroke="#10b981" strokeDasharray="4 2"
              label={{ value: "180 wpm (upper optimal)", position: "insideTopRight",
                       fontSize: 10, fill: "#10b981" }} />
            <ReferenceLine y={120} stroke="#10b981" strokeDasharray="4 2"
              label={{ value: "120 wpm (lower optimal)", position: "insideBottomRight",
                       fontSize: 10, fill: "#10b981" }} />
            <Bar dataKey="wpm" fill="#3b82f6" name="WPM" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Behavioral Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Response Time */}
        <div
          className="rounded-lg p-4"
          style={{
            background: "var(--background)",
            border: "1px solid var(--border)",
          }}
        >
          <h4
            className="text-sm font-medium mb-3"
            style={{ color: "var(--text-secondary)" }}
          >
            Response Time
          </h4>
          <div className="flex items-end justify-between">
            <div>
              <p
                className="text-3xl font-bold"
                style={{ color: "var(--text-primary)" }}
              >
                {ba.response_time_analysis?.avg_agent_response_time?.toFixed(
                  1,
                ) || "0"}
                <span
                  className="text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
                  s
                </span>
              </p>
              <p className="text-sm mt-1" style={{ color: "#0caf60" }}>
                {ba.response_time_analysis?.agent_response_assessment?.replace(
                  /_/g,
                  " ",
                ) || "N/A"}
              </p>
            </div>
            <Clock className="w-8 h-8" style={{ color: "var(--accent)" }} />
          </div>
        </div>

        {/* Interruptions */}
        <div
          className="rounded-lg p-4"
          style={{
            background: "var(--background)",
            border: "1px solid var(--border)",
          }}
        >
          <h4
            className="text-sm font-medium mb-3"
            style={{ color: "var(--text-secondary)" }}
          >
            Interruptions
          </h4>
          <div className="flex items-end justify-between">
            <div>
              <p
                className="text-3xl font-bold"
                style={{ color: "var(--text-primary)" }}
              >
                {ba.interruption_analysis?.interruption_count || 0}
              </p>
              <p className="text-sm mt-1" style={{ color: "#0caf60" }}>
                Agent: {ba.interruption_analysis?.agent_interruptions || 0},
                Customer:{" "}
                {ba.interruption_analysis?.customer_interruptions || 0}
              </p>
            </div>
            <AlertCircle className="w-8 h-8" style={{ color: "#e68a00" }} />
          </div>
        </div>

        {/* Questions */}
        <div
          className="rounded-lg p-4"
          style={{
            background: "var(--background)",
            border: "1px solid var(--border)",
          }}
        >
          <h4
            className="text-sm font-medium mb-3"
            style={{ color: "var(--text-secondary)" }}
          >
            Questions Asked
          </h4>
          <div className="flex items-end justify-between">
            <div>
              <p
                className="text-3xl font-bold"
                style={{ color: "var(--text-primary)" }}
              >
                {ba.question_analysis?.agent_questions || 0}
              </p>
              <p
                className="text-sm mt-1"
                style={{ color: "var(--text-secondary)" }}
              >
                Pattern:{" "}
                {ba.question_analysis?.question_pattern?.replace(/_/g, " ") ||
                  "N/A"}
              </p>
            </div>
            <MessageSquare
              className="w-8 h-8"
              style={{ color: "var(--accent)" }}
            />
          </div>
        </div>

        {/* Active Listening */}
        <div
          className="rounded-lg p-4"
          style={{
            background: "var(--background)",
            border: "1px solid var(--border)",
          }}
        >
          <h4
            className="text-sm font-medium mb-3"
            style={{ color: "var(--text-secondary)" }}
          >
            Active Listening
          </h4>
          <div className="flex items-end justify-between">
            <div>
              <p
                className="text-3xl font-bold"
                style={{ color: "var(--text-primary)" }}
              >
                {ba.active_listening?.acknowledgment_count || 0}
              </p>
              <p className="text-sm mt-1" style={{ color: "#0caf60" }}>
                {ba.active_listening?.listening_assessment?.replace(
                  /_/g,
                  " ",
                ) || "N/A"}
              </p>
            </div>
            <Heart className="w-8 h-8" style={{ color: "#ec4899" }} />
          </div>
        </div>
      </div>

      {/* Behavioral Summary Narrative */}
      {bs.summary && (
        <div
          className="rounded-lg p-6"
          style={{
            background: "var(--background)",
            border: "1px solid var(--border)",
          }}
        >
          <h3
            className="text-lg font-semibold mb-3 flex items-center"
            style={{ color: "var(--text-primary)" }}
          >
            <Activity
              className="w-5 h-5 mr-2"
              style={{ color: "var(--accent)" }}
            />
            Behavioral Summary
          </h3>
          <p
            className="leading-relaxed"
            style={{ color: "var(--text-secondary)" }}
          >
            {bs.summary}
          </p>
        </div>
      )}

      {/* Strengths and Areas for Improvement */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {strengths.length > 0 && (
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
              <CheckCircle
                className="w-5 h-5 mr-2"
                style={{ color: "#0caf60" }}
              />
              Strengths
            </h3>
            <ul className="space-y-2">
              {strengths.map((s: any, index: number) => (
                <li key={index} className="flex items-start">
                  <TrendingUp
                    className="w-4 h-4 mr-2 mt-0.5 flex-shrink-0"
                    style={{ color: "#0caf60" }}
                  />
                  <span style={{ color: "var(--text-secondary)" }}>
                    {typeof s === "string" ? s : s.description}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {issues.length > 0 && (
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
              <TrendingDown
                className="w-5 h-5 mr-2"
                style={{ color: "#e68a00" }}
              />
              Areas for Improvement
            </h3>
            <ul className="space-y-2">
              {issues.map((issue: any, index: number) => (
                <li key={index} className="flex items-start">
                  <AlertCircle
                    className="w-4 h-4 mr-2 mt-0.5 flex-shrink-0"
                    style={{ color: "#e68a00" }}
                  />
                  <span style={{ color: "var(--text-secondary)" }}>
                    {typeof issue === "string" ? issue : issue.description}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
