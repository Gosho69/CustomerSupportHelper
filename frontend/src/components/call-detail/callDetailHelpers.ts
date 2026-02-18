import React from "react";

export interface CallDetailModalProps {
  callId: number;
  onClose: () => void;
}

export interface CallDetail {
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

export const EMOTION_COLORS: Record<string, string> = {
  happy: "#10b981",
  frustrated: "#f59e0b",
  neutral: "#6b7280",
  sad: "#3b82f6",
  angry: "#ef4444",
  confused: "#8b5cf6",
};

export const formatDuration = (seconds: number | null | undefined): string => {
  if (!seconds || seconds === 0) return "0:00";
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
};

export const formatDate = (dateString: string): string => {
  if (!dateString) return "";
  const date = new Date(dateString);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
};

export const getSentimentColor = (sentiment: string): string => {
  switch (sentiment?.toLowerCase()) {
    case "positive":
      return "text-green-600";
    case "negative":
      return "text-red-600";
    default:
      return "text-gray-500";
  }
};

export const getScoreColor = (score: number): string => {
  if (score >= 80) return "text-green-600";
  if (score >= 60) return "text-yellow-600";
  return "text-red-600";
};
