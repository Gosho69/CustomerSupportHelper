"use client";

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import {
  X,
  Phone,
  Clock,
  Calendar,
  TrendingUp,
  TrendingDown,
  Smile,
  Frown,
  Meh,
  Heart,
  MessageSquare,
  Activity,
  Award,
  AlertCircle,
  CheckCircle,
  Lightbulb,
  BarChart3,
} from "lucide-react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import { api, callsApi } from "@/lib/api";

interface CallDetailModalProps {
  callId: number;
  onClose: () => void;
}

interface CallDetail {
  id: number;
  agent_name: string;
  call_date: string;
  duration: number;
  transcript: any;
  transcript_summary: any;
  emotional_analysis: any;
  emotional_summary: any;
  behavioral_analysis: any;
  behavioral_summary: any;
  coaching_tips: any;
  topic_analysis: any;
}

export default function CallDetailModal({
  callId,
  onClose,
}: CallDetailModalProps) {
  const [call, setCall] = useState<CallDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<
    "overview" | "transcript" | "emotions" | "behavior" | "coaching"
  >("overview");
  const [portalRoot, setPortalRoot] = useState<HTMLElement | null>(null);

  useEffect(() => {
    let root = document.getElementById("modal-portal");
    if (!root) {
      root = document.createElement("div");
      root.id = "modal-portal";
      document.body.appendChild(root);
    }
    setPortalRoot(root);

    return () => {
      if (root && root.childNodes.length === 0) {
        document.body.removeChild(root);
      }
    };
  }, []);

  useEffect(() => {
    fetchCallDetail();
  }, [callId]);

  const fetchCallDetail = async () => {
    try {
      setLoading(true);
      const response = await callsApi.getCallDetail(callId);
      setCall(response.data);
    } catch (error) {
      console.error("Failed to fetch call details:", error);
    } finally {
      setLoading(false);
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const getEmotionIcon = (emotion: string) => {
    switch (emotion.toLowerCase()) {
      case "happy":
        return <Smile className="w-5 h-5" style={{ color: "#0caf60" }} />;
      case "sad":
        return <Frown className="w-5 h-5" style={{ color: "var(--accent)" }} />;
      case "frustrated":
        return <AlertCircle className="w-5 h-5" style={{ color: "#e68a00" }} />;
      default:
        return (
          <Meh className="w-5 h-5" style={{ color: "var(--text-secondary)" }} />
        );
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment.toLowerCase()) {
      case "positive":
        return "text-green-600";
      case "negative":
        return "text-red-600";
      default:
        return "text-gray-500";
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return "text-green-600";
    if (score >= 60) return "text-yellow-600";
    return "text-red-600";
  };

  const EMOTION_COLORS = {
    happy: "#10b981",
    frustrated: "#f59e0b",
    neutral: "#6b7280",
    sad: "#3b82f6",
    angry: "#ef4444",
    confused: "#8b5cf6",
  };

  const renderOverview = () => {
    if (!call) return null;

    const emotionData = call.emotional_summary?.emotion_distribution
      ? Object.entries(call.emotional_summary.emotion_distribution).map(
          ([name, value]) => ({
            name: name.charAt(0).toUpperCase() + name.slice(1),
            value,
          }),
        )
      : [];

    const behaviorRadarData = [
      {
        metric: "Response Time",
        score:
          call.behavioral_analysis?.response_time_analysis
            ?.agent_avg_response_time <= 1.5
            ? 95
            : 70,
      },
      {
        metric: "Speaking Pace",
        score:
          call.behavioral_analysis?.words_per_minute?.agent_wpm >= 140 &&
          call.behavioral_analysis?.words_per_minute?.agent_wpm <= 160
            ? 90
            : 75,
      },
      {
        metric: "Active Listening",
        score: call.behavioral_analysis?.active_listening?.acknowledgment_count
          ? call.behavioral_analysis.active_listening.acknowledgment_count * 15
          : 80,
      },
      { metric: "Interruptions", score: 95 },
      {
        metric: "Questions",
        score:
          (call.behavioral_analysis?.question_analysis?.agent_questions || 0) *
          20,
      },
    ];

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
                <p
                  className="text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Overall Score
                </p>
                <p
                  className={`text-3xl font-bold mt-1 ${getScoreColor(
                    call.behavioral_analysis?.behavioral_score || 0,
                  )}`}
                >
                  {call.behavioral_analysis?.behavioral_score || 0}
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
                <p
                  className="text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Sentiment
                </p>
                <p
                  className={`text-xl font-bold mt-1 capitalize ${getSentimentColor(
                    call.emotional_summary?.overall_sentiment || "neutral",
                  )}`}
                >
                  {call.emotional_summary?.overall_sentiment || "Neutral"}
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
                <p
                  className="text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
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
                        EMOTION_COLORS[
                          entry.name.toLowerCase() as keyof typeof EMOTION_COLORS
                        ] || "#6b7280"
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
            {call.transcript_summary?.summary ||
              "No summary available for this call."}
          </p>
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
      </div>
    );
  };

  const renderTranscript = () => {
    if (!call) return null;

    return (
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
          Full Transcript
        </h3>
        <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2">
          {call.transcript?.text ? (
            <div
              className="whitespace-pre-wrap leading-relaxed"
              style={{ color: "var(--text-secondary)" }}
            >
              {call.transcript.text}
            </div>
          ) : (
            <p style={{ color: "var(--text-secondary)" }}>
              No transcript available
            </p>
          )}
        </div>
      </div>
    );
  };

  const renderEmotions = () => {
    if (!call) return null;

    const emotionTimelineData =
      call.emotional_analysis?.emotion_timeline?.map((item: any) => ({
        time: `${item.time}s`,
        emotion: item.emotion,
        speaker: item.speaker,
      })) || [];

    return (
      <div className="space-y-6">
        {/* Emotion Timeline */}
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
          <div className="space-y-4">
            {call.emotional_summary?.key_emotional_moments?.map(
              (moment: string, index: number) => (
                <div
                  key={index}
                  className="flex items-start space-x-3 p-3 rounded-lg"
                  style={{ background: "var(--accent-bg)" }}
                >
                  <div
                    className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
                    style={{ background: "var(--accent-bg)" }}
                  >
                    <span
                      className="text-sm font-medium"
                      style={{ color: "var(--accent)" }}
                    >
                      {index + 1}
                    </span>
                  </div>
                  <p
                    className="flex-1"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {moment}
                  </p>
                </div>
              ),
            )}
          </div>
        </div>

        {/* Detailed Turn Analysis */}
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
            {call.emotional_analysis?.turns?.map((turn: any, index: number) => (
              <div
                key={index}
                className="flex items-center justify-between p-3 rounded-lg"
                style={{ background: "var(--accent-bg)" }}
              >
                <div className="flex items-center space-x-3">
                  <div
                    className={`px-3 py-1 rounded-full text-xs font-medium ${
                      turn.speaker === "agent"
                        ? "bg-blue-50 text-blue-600"
                        : "bg-green-50 text-green-600"
                    }`}
                  >
                    {turn.speaker.toUpperCase()}
                  </div>
                  {getEmotionIcon(turn.emotion)}
                  <span
                    className="capitalize"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {turn.emotion}
                  </span>
                </div>
                <div className="flex items-center space-x-4">
                  <span
                    className={`text-sm font-medium ${getSentimentColor(
                      turn.sentiment,
                    )}`}
                  >
                    {turn.sentiment}
                  </span>
                  <span
                    className="text-sm"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {(turn.confidence * 100).toFixed(0)}% confidence
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const renderBehavior = () => {
    if (!call) return null;

    const wpmData = [
      {
        name: "Agent",
        wpm: call.behavioral_analysis?.words_per_minute?.agent_wpm || 0,
        optimal: 150,
      },
      {
        name: "Customer",
        wpm: call.behavioral_analysis?.words_per_minute?.customer_wpm || 0,
        optimal: 150,
      },
    ];

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
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={wpmData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ee" />
              <XAxis dataKey="name" tick={{ fill: "#697386" }} />
              <YAxis tick={{ fill: "#697386" }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#ffffff",
                  border: "1px solid var(--border)",
                  borderRadius: "0.5rem",
                  color: "var(--text-primary)",
                }}
              />
              <Legend />
              <Bar dataKey="wpm" fill="#3b82f6" name="Actual WPM" />
              <Bar dataKey="optimal" fill="#10b981" name="Optimal WPM" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Behavioral Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
                  {call.behavioral_analysis?.response_time_analysis?.agent_avg_response_time?.toFixed(
                    1,
                  ) || 0}
                  <span
                    className="text-sm"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    s
                  </span>
                </p>
                <p className="text-sm mt-1" style={{ color: "#0caf60" }}>
                  {call.behavioral_analysis?.response_time_analysis
                    ?.agent_assessment || "Good"}
                </p>
              </div>
              <Clock className="w-8 h-8" style={{ color: "var(--accent)" }} />
            </div>
          </div>

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
                  {call.behavioral_analysis?.interruption_analysis
                    ?.total_interruptions || 0}
                </p>
                <p className="text-sm mt-1" style={{ color: "#0caf60" }}>
                  {call.behavioral_analysis?.interruption_analysis
                    ?.assessment || "Excellent"}
                </p>
              </div>
              <AlertCircle className="w-8 h-8" style={{ color: "#e68a00" }} />
            </div>
          </div>

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
                  {call.behavioral_analysis?.question_analysis
                    ?.agent_questions || 0}
                </p>
                <p
                  className="text-sm mt-1"
                  style={{ color: "var(--text-secondary)" }}
                >
                  {call.behavioral_analysis?.question_analysis
                    ?.agent_open_questions || 0}{" "}
                  open,{" "}
                  {call.behavioral_analysis?.question_analysis
                    ?.agent_closed_questions || 0}{" "}
                  closed
                </p>
              </div>
              <MessageSquare
                className="w-8 h-8"
                style={{ color: "var(--accent)" }}
              />
            </div>
          </div>

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
                  {call.behavioral_analysis?.active_listening
                    ?.acknowledgment_count || 0}
                </p>
                <p className="text-sm mt-1" style={{ color: "#0caf60" }}>
                  {call.behavioral_analysis?.active_listening?.assessment ||
                    "Good"}
                </p>
              </div>
              <Heart className="w-8 h-8" style={{ color: "#ec4899" }} />
            </div>
          </div>
        </div>

        {/* Strengths and Areas for Improvement */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
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
              {call.behavioral_summary?.strengths?.map(
                (strength: string, index: number) => (
                  <li key={index} className="flex items-start">
                    <TrendingUp
                      className="w-4 h-4 mr-2 mt-0.5 flex-shrink-0"
                      style={{ color: "#0caf60" }}
                    />
                    <span style={{ color: "var(--text-secondary)" }}>
                      {strength}
                    </span>
                  </li>
                ),
              )}
            </ul>
          </div>

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
              {call.behavioral_summary?.areas_for_improvement?.map(
                (area: string, index: number) => (
                  <li key={index} className="flex items-start">
                    <AlertCircle
                      className="w-4 h-4 mr-2 mt-0.5 flex-shrink-0"
                      style={{ color: "#e68a00" }}
                    />
                    <span style={{ color: "var(--text-secondary)" }}>
                      {area}
                    </span>
                  </li>
                ),
              )}
            </ul>
          </div>
        </div>
      </div>
    );
  };

  const renderCoaching = () => {
    if (!call) return null;

    const positiveTips =
      call.coaching_tips?.tips?.filter(
        (tip: any) => tip.priority === "positive",
      ) || [];
    const improvementTips =
      call.coaching_tips?.tips?.filter(
        (tip: any) => tip.priority === "improvement",
      ) || [];

    return (
      <div className="space-y-6">
        {/* Positive Feedback */}
        {positiveTips.length > 0 && (
          <div
            className="rounded-lg p-6"
            style={{
              background: "var(--background)",
              border: "1px solid rgba(16,185,129,0.2)",
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
              What You Did Well
            </h3>
            <div className="space-y-4">
              {positiveTips.map((tip: any, index: number) => (
                <div
                  key={index}
                  className="flex items-start space-x-3 p-4 bg-green-500/10 border border-green-500/20 rounded-lg"
                >
                  <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                    <Award className="w-5 h-5" style={{ color: "#0caf60" }} />
                  </div>
                  <div className="flex-1">
                    <h4
                      className="text-sm font-medium mb-1"
                      style={{ color: "#0caf60" }}
                    >
                      {tip.category}
                    </h4>
                    <p style={{ color: "var(--text-secondary)" }}>{tip.tip}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Improvement Areas */}
        {improvementTips.length > 0 && (
          <div
            className="rounded-lg p-6"
            style={{
              background: "var(--background)",
              border: "1px solid rgba(245,158,11,0.2)",
            }}
          >
            <h3
              className="text-lg font-semibold mb-4 flex items-center"
              style={{ color: "var(--text-primary)" }}
            >
              <Lightbulb
                className="w-5 h-5 mr-2"
                style={{ color: "#e68a00" }}
              />
              Areas to Improve
            </h3>
            <div className="space-y-4">
              {improvementTips.map((tip: any, index: number) => (
                <div
                  key={index}
                  className="flex items-start space-x-3 p-4 bg-orange-500/10 border border-orange-500/20 rounded-lg"
                >
                  <div className="w-10 h-10 bg-orange-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                    <TrendingUp
                      className="w-5 h-5"
                      style={{ color: "#e68a00" }}
                    />
                  </div>
                  <div className="flex-1">
                    <h4
                      className="text-sm font-medium mb-1"
                      style={{ color: "#e68a00" }}
                    >
                      {tip.category}
                    </h4>
                    <p style={{ color: "var(--text-secondary)" }}>{tip.tip}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Topics Discussed */}
        {call.topic_analysis?.main_topics && (
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
              Topics Discussed
            </h3>
            <div className="flex flex-wrap gap-3">
              {call.topic_analysis.main_topics.map(
                (topic: any, index: number) => (
                  <div
                    key={index}
                    className="px-4 py-2 bg-blue-50 border border-blue-100 rounded-lg"
                  >
                    <span
                      className="font-medium"
                      style={{ color: "var(--accent)" }}
                    >
                      {topic.topic}
                    </span>
                    <span
                      className="text-sm ml-2"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {(topic.relevance * 100).toFixed(0)}%
                    </span>
                  </div>
                ),
              )}
            </div>
          </div>
        )}
      </div>
    );
  };

  if (!portalRoot) return null;

  return createPortal(
    <div
      className="fixed inset-0 bg-black/20 z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-lg w-full max-w-6xl max-h-[90vh] overflow-hidden flex flex-col shadow-xl"
        style={{ border: "1px solid var(--border)" }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          className="p-6 flex items-center justify-between"
          style={{ borderBottom: "1px solid var(--border)" }}
        >
          <div className="flex items-center space-x-4">
            <div
              className="w-12 h-12 rounded-lg flex items-center justify-center"
              style={{ background: "var(--accent-bg)" }}
            >
              <Phone className="w-6 h-6" style={{ color: "var(--accent)" }} />
            </div>
            <div>
              <h2
                className="text-2xl font-bold"
                style={{ color: "var(--text-primary)" }}
              >
                Call #{callId} Details
              </h2>
              {call && (
                <div
                  className="flex items-center space-x-4 mt-1 text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
                  <span className="flex items-center">
                    <Calendar className="w-4 h-4 mr-1" />
                    {formatDate(call.call_date)}
                  </span>
                  <span className="flex items-center">
                    <Clock className="w-4 h-4 mr-1" />
                    {formatDuration(call.duration)}
                  </span>
                </div>
              )}
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X className="w-6 h-6" style={{ color: "var(--text-tertiary)" }} />
          </button>
        </div>

        {/* Tabs */}
        <div
          className="px-6 pt-4"
          style={{ borderBottom: "1px solid var(--border)" }}
        >
          <div className="flex space-x-1">
            {[
              { id: "overview", label: "Overview" },
              { id: "transcript", label: "Transcript" },
              { id: "emotions", label: "Emotions" },
              { id: "behavior", label: "Behavior" },
              { id: "coaching", label: "Coaching" },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() =>
                  setActiveTab(
                    tab.id as
                      | "overview"
                      | "transcript"
                      | "emotions"
                      | "behavior"
                      | "coaching",
                  )
                }
                className="px-6 py-3 rounded-t-lg font-medium transition-all"
                style={{
                  color:
                    activeTab === tab.id
                      ? "var(--accent)"
                      : "var(--text-secondary)",
                  borderBottom:
                    activeTab === tab.id
                      ? "2px solid var(--accent)"
                      : "2px solid transparent",
                  background:
                    activeTab === tab.id ? "var(--accent-bg)" : "transparent",
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
            </div>
          ) : (
            <>
              {activeTab === "overview" && renderOverview()}
              {activeTab === "transcript" && renderTranscript()}
              {activeTab === "emotions" && renderEmotions()}
              {activeTab === "behavior" && renderBehavior()}
              {activeTab === "coaching" && renderCoaching()}
            </>
          )}
        </div>
      </div>
    </div>,
    portalRoot,
  );
}
