"use client";

import {
  Smile,
  Frown,
  Meh,
  AlertCircle,
  Activity,
  TrendingUp,
} from "lucide-react";
import {
  CallDetail,
  formatDuration,
  getSentimentColor,
} from "./callDetailHelpers";

interface EmotionsTabProps {
  call: CallDetail;
}

const getEmotionIcon = (emotion: string) => {
  switch (emotion?.toLowerCase()) {
    case "happy":
      return <Smile className="w-5 h-5" style={{ color: "#0caf60" }} />;
    case "sad":
      return <Frown className="w-5 h-5" style={{ color: "var(--accent)" }} />;
    case "frustrated":
    case "angry":
      return <AlertCircle className="w-5 h-5" style={{ color: "#e68a00" }} />;
    case "fearful":
      return <AlertCircle className="w-5 h-5" style={{ color: "#f59e0b" }} />;
    case "surprised":
      return <Meh className="w-5 h-5" style={{ color: "#8b5cf6" }} />;
    default:
      return (
        <Meh className="w-5 h-5" style={{ color: "var(--text-secondary)" }} />
      );
  }
};

export default function EmotionsTab({ call }: EmotionsTabProps) {
  // Backend: emotional_summary.key_moments (NOT key_emotional_moments)
  const keyMoments = call.emotional_summary?.key_moments || [];

  // Backend: emotional_analysis is a FLAT array of Turn dicts (NOT nested under .turns)
  const turns: any[] = Array.isArray(call.emotional_analysis)
    ? call.emotional_analysis
    : call.emotional_analysis?.turns || [];

  // Backend: emotional_summary.emotional_journey
  const journey = call.emotional_summary?.emotional_journey;

  return (
    <div className="space-y-6">
      {/* Emotional Journey */}
      {journey && (
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
            <TrendingUp
              className="w-5 h-5 mr-2"
              style={{ color: "var(--accent)" }}
            />
            Emotional Journey
          </h3>
          <div className="space-y-3">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                {getEmotionIcon(journey.start_emotion)}
                <span
                  className="capitalize font-medium"
                  style={{ color: "var(--text-primary)" }}
                >
                  {journey.start_emotion}
                </span>
              </div>
              <div
                className="flex-1 h-1 rounded-full"
                style={{ background: "var(--border)" }}
              >
                <div
                  className="h-full rounded-full"
                  style={{
                    width: "100%",
                    background:
                      journey.trajectory === "resolved"
                        ? "#10b981"
                        : journey.trajectory === "escalated"
                          ? "#ef4444"
                          : "var(--accent)",
                  }}
                />
              </div>
              <div className="flex items-center space-x-2">
                {getEmotionIcon(journey.end_emotion)}
                <span
                  className="capitalize font-medium"
                  style={{ color: "var(--text-primary)" }}
                >
                  {journey.end_emotion}
                </span>
              </div>
            </div>
            <p
              className="text-sm text-center capitalize"
              style={{ color: "var(--text-secondary)" }}
            >
              Trajectory: {journey.trajectory?.replace(/_/g, " ")}
            </p>
            {journey.description && (
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                {journey.description}
              </p>
            )}
          </div>
        </div>
      )}

      {/* Key Emotional Moments */}
      {keyMoments.length > 0 && (
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
            <AlertCircle
              className="w-5 h-5 mr-2"
              style={{ color: "#e68a00" }}
            />
            Key Emotional Moments
          </h3>
          <div className="space-y-3">
            {keyMoments.map((moment: any, index: number) => (
              <div
                key={index}
                className="flex items-start space-x-3 p-3 rounded-lg"
                style={{ background: "var(--accent-bg)" }}
              >
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                    moment.type === "escalation"
                      ? "bg-red-100"
                      : moment.type === "apology"
                        ? "bg-blue-100"
                        : moment.type === "empathy"
                          ? "bg-purple-100"
                          : "bg-green-100"
                  }`}
                >
                  <span
                    className={`text-xs font-bold ${
                      moment.type === "escalation"
                        ? "text-red-600"
                        : moment.type === "apology"
                          ? "text-blue-600"
                          : moment.type === "empathy"
                            ? "text-purple-600"
                            : "text-green-600"
                    }`}
                  >
                    {moment.type === "escalation"
                      ? "!"
                      : moment.type === "apology"
                        ? "♥"
                        : moment.type === "empathy"
                          ? "☺"
                          : "✓"}
                  </span>
                </div>
                <div className="flex-1">
                  <div className="flex items-center space-x-2 mb-1">
                    <span
                      className="text-xs font-semibold capitalize px-2 py-0.5 rounded-full"
                      style={{
                        background:
                          moment.type === "escalation" ? "#fef2f2" : "#f0fdf4",
                        color:
                          moment.type === "escalation" ? "#dc2626" : "#16a34a",
                      }}
                    >
                      {moment.type}
                    </span>
                    <span
                      className="text-xs"
                      style={{ color: "var(--text-tertiary)" }}
                    >
                      {moment.speaker} at{" "}
                      {formatDuration(Math.round(moment.timestamp_sec || 0))}
                    </span>
                  </div>
                  <p
                    className="text-sm"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {moment.text}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Turn-by-Turn Emotions */}
      {turns.length > 0 && (
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
            Turn-by-Turn Emotions
          </h3>
          <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2">
            {turns.map((turn: any, index: number) => (
              <div
                key={index}
                className="flex items-center justify-between p-3 rounded-lg"
                style={{ background: "var(--accent-bg)" }}
              >
                <div className="flex items-center space-x-3">
                  <div
                    className={`px-3 py-1 rounded-full text-xs font-medium ${
                      turn.speaker?.toLowerCase() === "agent"
                        ? "bg-blue-50 text-blue-600"
                        : "bg-green-50 text-green-600"
                    }`}
                  >
                    {(turn.speaker || "Unknown").toUpperCase()}
                  </div>
                  {getEmotionIcon(turn.emotion)}
                  <span
                    className="capitalize"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {turn.emotion || "neutral"}
                  </span>
                </div>
                <div className="flex items-center space-x-4">
                  <span
                    className={`text-sm font-medium ${getSentimentColor(
                      turn.sentiment,
                    )}`}
                  >
                    {turn.sentiment || "neutral"}
                  </span>
                  {turn.emotion_score != null && (
                    <span
                      className="text-sm"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {(turn.emotion_score * 100).toFixed(0)}% confidence
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {turns.length === 0 && keyMoments.length === 0 && !journey && (
        <div
          className="rounded-lg p-12 text-center"
          style={{
            background: "var(--background)",
            border: "1px solid var(--border)",
          }}
        >
          <p style={{ color: "var(--text-secondary)" }}>
            No emotion analysis data available
          </p>
        </div>
      )}
    </div>
  );
}
