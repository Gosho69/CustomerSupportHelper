"use client";

import { useState, useEffect } from "react";
import { UploadCall } from "@/components/calls";
import AnalysisProgress from "@/components/calls/analysis-progress/AnalysisProgress";
import { useAnalysisStore } from "@/store/analysisStore";
import { isAnalysisInProgress } from "@/lib/activeAnalysis";

export default function UploadCallPage() {
  const { files } = useAnalysisStore();
  // null = not yet determined (avoid SSR/client mismatch)
  const [mode, setMode] = useState<"upload" | "analysis" | null>(null);

  // Determine initial mode on first client render
  useEffect(() => {
    setMode(files.length > 0 || isAnalysisInProgress() ? "analysis" : "upload");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // When UploadCall puts files in the store, switch to analysis view
  useEffect(() => {
    if (files.length > 0) setMode("analysis");
  }, [files.length]);

  if (mode === null) return null;

  if (mode === "analysis") {
    return <AnalysisProgress onDone={() => setMode("upload")} />;
  }

  return <UploadCall />;
}
