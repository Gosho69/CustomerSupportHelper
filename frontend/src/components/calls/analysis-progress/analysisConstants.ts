import React from "react";

export interface AnalysisStep {
  id: string;
  label: string;
  description: string;
  icon: React.ReactNode;
  status: "pending" | "active" | "completed" | "error";
}

export interface FileAnalysisState {
  fileName: string;
  fileSize: number;
  currentStepIndex: number;
  status: "processing" | "completed" | "error";
  error?: string;
  callId?: number;
}

// Time intervals between steps to simulate progress while the API processes
export const STEP_INTERVALS = [3000, 8000, 6000, 5000, 5000, 4000, 4000, 0];

export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
};
