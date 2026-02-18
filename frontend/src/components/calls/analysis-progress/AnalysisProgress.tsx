"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  Loader2,
  CheckCircle,
  AlertCircle,
  ArrowLeft,
  Eye,
} from "lucide-react";
import { callsApi } from "@/lib/api";
import { useAnalysisStore } from "@/store/analysisStore";
import CallDetailModal from "@/components/call-detail";
import { FileAnalysisState, STEP_INTERVALS } from "./analysisConstants";
import { ANALYSIS_STEPS } from "./AnalysisSteps";
import FileProgressCard from "./FileProgressCard";

export default function AnalysisProgress() {
  const router = useRouter();
  const { files, summarizationModel, clearFiles } = useAnalysisStore();
  const [fileStates, setFileStates] = useState<FileAnalysisState[]>([]);
  const [overallStatus, setOverallStatus] = useState<
    "processing" | "completed" | "error"
  >("processing");
  const [viewingCallId, setViewingCallId] = useState<number | null>(null);
  const processingRef = useRef(false);
  const stepTimersRef = useRef<NodeJS.Timeout[]>([]);
  const hasAutoOpenedRef = useRef(false);

  // Redirect if no files
  useEffect(() => {
    if (files.length === 0) {
      router.replace("/dashboard/upload-call");
    }
  }, [files, router]);

  // Initialize file states
  useEffect(() => {
    if (files.length > 0 && fileStates.length === 0) {
      setFileStates(
        files.map((f) => ({
          fileName: f.name,
          fileSize: f.size,
          currentStepIndex: 0,
          status: "processing" as const,
        })),
      );
    }
  }, [files, fileStates.length]);

  // Advance step for a file with simulated timing
  const advanceSteps = useCallback((fileIndex: number) => {
    let currentStep = 0;

    const advanceNext = () => {
      if (currentStep < ANALYSIS_STEPS.length - 1) {
        currentStep++;
        setFileStates((prev) => {
          const next = [...prev];
          if (next[fileIndex] && next[fileIndex].status === "processing") {
            next[fileIndex] = {
              ...next[fileIndex],
              currentStepIndex: currentStep,
            };
          }
          return next;
        });

        if (currentStep < ANALYSIS_STEPS.length - 1) {
          const timer = setTimeout(advanceNext, STEP_INTERVALS[currentStep]);
          stepTimersRef.current.push(timer);
        }
      }
    };

    const timer = setTimeout(advanceNext, STEP_INTERVALS[0]);
    stepTimersRef.current.push(timer);
  }, []);

  // Complete all steps for a file (when API returns)
  const completeFile = useCallback((fileIndex: number, callId?: number) => {
    setFileStates((prev) => {
      const next = [...prev];
      next[fileIndex] = {
        ...next[fileIndex],
        currentStepIndex: ANALYSIS_STEPS.length - 1,
        status: "completed",
        callId,
      };
      return next;
    });
  }, []);

  const failFile = useCallback((fileIndex: number, error: string) => {
    setFileStates((prev) => {
      const next = [...prev];
      next[fileIndex] = {
        ...next[fileIndex],
        status: "error",
        error,
      };
      return next;
    });
  }, []);

  // Process all files
  useEffect(() => {
    if (files.length === 0 || fileStates.length === 0 || processingRef.current)
      return;
    processingRef.current = true;

    const processFiles = async () => {
      const promises = files.map(async (analysisFile, index) => {
        advanceSteps(index);

        try {
          const formData = new FormData();
          formData.append("audio_file", analysisFile.file);
          formData.append("summarization_model", summarizationModel);
          const response = await callsApi.uploadCall(formData);
          const callId = response.data?.call?.id;
          completeFile(index, callId);
        } catch (error: any) {
          const message =
            error.response?.data?.error || error.message || "Analysis failed";
          failFile(index, message);
        }
      });

      await Promise.all(promises);

      setFileStates((prev) => {
        const allDone = prev.every(
          (f) => f.status === "completed" || f.status === "error",
        );
        if (allDone) {
          const allError = prev.every((f) => f.status === "error");
          setOverallStatus(allError ? "error" : "completed");
        }
        return prev;
      });
    };

    processFiles();

    return () => {
      stepTimersRef.current.forEach(clearTimeout);
    };
  }, [files, fileStates.length, advanceSteps, completeFile, failFile]);

  // Auto-open CallDetailModal when a single file completes and auto-close when analysis is done
  useEffect(() => {
    const completedFiles = fileStates.filter(
      (f) => f.status === "completed" && f.callId,
    );
    if (
      completedFiles.length === 1 &&
      fileStates.length === 1 &&
      viewingCallId === null &&
      !hasAutoOpenedRef.current
    ) {
      setViewingCallId(completedFiles[0].callId!);
      hasAutoOpenedRef.current = true;
    }
  }, [fileStates, viewingCallId]);

  // Auto-return to upload page when analysis is complete
  useEffect(() => {
    if (overallStatus === "completed") {
      handleBackToUpload();
    }
  }, [overallStatus]);

  const handleBackToUpload = () => {
    clearFiles();
    router.push("/dashboard/upload-call");
  };

  if (files.length === 0) {
    return null;
  }

  const completedCount = fileStates.filter(
    (f) => f.status === "completed",
  ).length;
  const errorCount = fileStates.filter((f) => f.status === "error").length;
  const totalCount = fileStates.length;
  const allDone = completedCount + errorCount === totalCount && totalCount > 0;

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Header */}
      <div
        className="rounded-lg p-8"
        style={{ background: "#ffffff", border: "1px solid var(--border)" }}
      >
        <div className="flex items-center justify-between">
          <div>
            <h1
              className="text-3xl font-bold mb-2"
              style={{ color: "var(--text-primary)" }}
            >
              {allDone ? "Analysis Complete" : "Analyzing Call"}
            </h1>
            <p style={{ color: "var(--text-secondary)" }}>
              {allDone
                ? `${completedCount} of ${totalCount} file${totalCount > 1 ? "s" : ""} analyzed successfully`
                : `Processing ${totalCount} file${totalCount > 1 ? "s" : ""}... This may take a few minutes.`}
            </p>
          </div>
          {!allDone && (
            <div className="flex items-center space-x-3">
              <div
                className="w-12 h-12 rounded-full flex items-center justify-center"
                style={{ background: "var(--accent-bg)" }}
              >
                <Loader2
                  className="w-6 h-6 animate-spin"
                  style={{ color: "var(--accent)" }}
                />
              </div>
            </div>
          )}
          {allDone && (
            <div className="flex items-center space-x-3">
              <div
                className={`w-12 h-12 rounded-full flex items-center justify-center ${
                  errorCount === totalCount ? "bg-red-50" : "bg-green-50"
                }`}
              >
                {errorCount === totalCount ? (
                  <AlertCircle className="w-6 h-6 text-red-500" />
                ) : (
                  <CheckCircle className="w-6 h-6 text-green-500" />
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Overall progress bar */}
      {!allDone && (
        <div
          className="rounded-lg p-6"
          style={{ background: "#ffffff", border: "1px solid var(--border)" }}
        >
          <div className="flex items-center justify-between mb-3">
            <span
              className="text-sm font-medium"
              style={{ color: "var(--text-secondary)" }}
            >
              Overall Progress
            </span>
            <span
              className="text-sm font-medium"
              style={{ color: "var(--text-secondary)" }}
            >
              {completedCount} / {totalCount} complete
            </span>
          </div>
          <div className="w-full bg-gray-100 rounded-full h-3 overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-700 ease-out"
              style={{
                width: `${totalCount > 0 ? (completedCount / totalCount) * 100 : 0}%`,
                background: "var(--accent)",
              }}
            />
          </div>
        </div>
      )}

      {/* File progress cards */}
      <div className="space-y-4">
        {fileStates.map((fileState, fileIndex) => (
          <FileProgressCard
            key={fileIndex}
            fileState={fileState}
            onViewCall={(callId) => setViewingCallId(callId)}
          />
        ))}
      </div>

      {/* Actions when complete */}
      {allDone && (
        <div
          className="rounded-lg p-6 flex items-center justify-between"
          style={{ background: "#ffffff", border: "1px solid var(--border)" }}
        >
          <button
            onClick={handleBackToUpload}
            className="flex items-center space-x-2 px-5 py-2.5 rounded-lg font-medium transition-colors hover:opacity-80"
            style={{
              background: "var(--background)",
              color: "var(--text-primary)",
              border: "1px solid var(--border)",
            }}
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Upload More</span>
          </button>

          {completedCount > 0 && (
            <button
              onClick={() => router.push("/dashboard/calls")}
              className="flex items-center space-x-2 px-5 py-2.5 rounded-lg font-medium transition-colors hover:opacity-80"
              style={{ background: "var(--accent)", color: "#ffffff" }}
            >
              <Eye className="w-4 h-4" />
              <span>View My Calls</span>
            </button>
          )}
        </div>
      )}

      {/* Call Detail Modal */}
      {viewingCallId && (
        <CallDetailModal
          callId={viewingCallId}
          onClose={() => setViewingCallId(null)}
        />
      )}
    </div>
  );
}
