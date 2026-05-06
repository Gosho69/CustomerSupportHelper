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
import {
  markAnalysisStarted,
  addStoredCall,
  getStoredCallId,
  getStoredStep,
  updateStoredCallStep,
  getActiveAnalysis,
  clearActiveAnalysis,
} from "@/lib/activeAnalysis";
import CallDetailModal from "@/components/call-detail";
import { FileAnalysisState, STEP_INTERVALS } from "./analysisConstants";
import { ANALYSIS_STEPS } from "./AnalysisSteps";
import FileProgressCard from "./FileProgressCard";

interface Props {
  /** Called when analysis finishes or user wants to upload more.
   *  If not provided (standalone /analysis route) falls back to router navigation. */
  onDone?: () => void;
}

export default function AnalysisProgress({ onDone }: Props) {
  const router = useRouter();
  const { files, summarizationModel, clearFiles } = useAnalysisStore();
  const [fileStates, setFileStates] = useState<FileAnalysisState[]>([]);
  const [overallStatus, setOverallStatus] = useState<
    "processing" | "completed" | "error"
  >("processing");
  const [viewingCallId, setViewingCallId] = useState<number | null>(null);
  const [isRecoveryMode, setIsRecoveryMode] = useState(false);
  const processingRef = useRef(false);
  const stepTimersRef = useRef<NodeJS.Timeout[]>([]);
  const pollTimersRef = useRef<NodeJS.Timeout[]>([]);
  const hasAutoOpenedRef = useRef(false);
  const modeCheckedRef = useRef(false);

  // On mount only: decide if we need recovery mode or should exit
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (modeCheckedRef.current) return;
    modeCheckedRef.current = true;

    if (files.length > 0) return; // normal mode — files are already in the store

    const active = getActiveAnalysis();
    if (active?.inProgress && active.calls.length > 0) {
      setIsRecoveryMode(true);
    } else {
      // Nothing to show — hand control back to the parent (or navigate away)
      if (onDone) onDone();
      else router.replace("/dashboard/upload-call");
    }
  });

  // Initialize file states — normal mode (restore saved step if available)
  useEffect(() => {
    if (files.length === 0 || fileStates.length > 0 || isRecoveryMode) return;
    setFileStates(
      files.map((f) => ({
        fileName: f.name,
        fileSize: f.size,
        currentStepIndex: getStoredStep(f.name) ?? 0,
        status: "processing" as const,
      })),
    );
  }, [files, fileStates.length, isRecoveryMode]);

  // Initialize file states — recovery mode (restore from localStorage)
  useEffect(() => {
    if (!isRecoveryMode || fileStates.length > 0) return;
    const active = getActiveAnalysis();
    if (!active?.calls.length) return;
    setFileStates(
      active.calls.map((c) => ({
        fileName: c.fileName,
        fileSize: c.fileSize,
        currentStepIndex: c.currentStepIndex ?? 2,
        status: "processing" as const,
      })),
    );
  }, [isRecoveryMode, fileStates.length]);

  // Advance simulated steps, persisting each step to localStorage
  const advanceSteps = useCallback(
    (fileIndex: number, startStep = 0, fileName?: string) => {
      let currentStep = startStep;

      const advanceNext = () => {
        if (currentStep >= ANALYSIS_STEPS.length - 1) return;
        currentStep++;
        setFileStates((prev) => {
          const next = [...prev];
          if (next[fileIndex]?.status === "processing") {
            next[fileIndex] = { ...next[fileIndex], currentStepIndex: currentStep };
          }
          return next;
        });
        if (fileName) updateStoredCallStep(fileName, currentStep);
        if (currentStep < ANALYSIS_STEPS.length - 1) {
          const t = setTimeout(advanceNext, STEP_INTERVALS[currentStep] ?? 4000);
          stepTimersRef.current.push(t);
        }
      };

      const t = setTimeout(advanceNext, STEP_INTERVALS[startStep] ?? 3000);
      stepTimersRef.current.push(t);
    },
    [],
  );

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
      next[fileIndex] = { ...next[fileIndex], status: "error", error };
      return next;
    });
  }, []);

  // Poll backend until a call is complete or failed
  const pollCallStatus = useCallback(
    (callId: number, fileIndex: number): Promise<void> =>
      new Promise<void>((resolve) => {
        const POLL_MS = 5_000;
        const TIMEOUT_MS = 35 * 60 * 1_000;
        const startedAt = Date.now();

        const id = setInterval(async () => {
          if (Date.now() - startedAt > TIMEOUT_MS) {
            clearInterval(id);
            failFile(
              fileIndex,
              "Analysis timed out. The server may have run out of memory — please try again or contact support.",
            );
            resolve();
            return;
          }
          try {
            const res = await callsApi.getCallStatus(callId);
            const { status: s, error_message } = res.data;
            if (s === "completed") {
              clearInterval(id);
              completeFile(fileIndex, callId);
              resolve();
            } else if (s === "failed") {
              clearInterval(id);
              failFile(fileIndex, error_message || "Analysis failed");
              resolve();
            }
          } catch {
            clearInterval(id);
            failFile(fileIndex, "Failed to check analysis status");
            resolve();
          }
        }, POLL_MS);

        pollTimersRef.current.push(id as unknown as NodeJS.Timeout);
      }),
    [completeFile, failFile],
  );

  // Finish handler — updates overall status and clears persistence
  const handleAllDone = useCallback((states: FileAnalysisState[]) => {
    const allError = states.every((f) => f.status === "error");
    setOverallStatus(allError ? "error" : "completed");
    clearActiveAnalysis();
  }, []);

  // Process files — normal mode
  useEffect(() => {
    if (
      files.length === 0 ||
      fileStates.length === 0 ||
      processingRef.current ||
      isRecoveryMode
    )
      return;
    processingRef.current = true;
    markAnalysisStarted();

    const run = async () => {
      await Promise.all(
        files.map(async (f, i) => {
          const savedStep = getStoredStep(f.name) ?? 0;
          advanceSteps(i, savedStep, f.name);
          try {
            const existingId = getStoredCallId(f.name);
            let callId: number;
            if (existingId !== null) {
              callId = existingId;
            } else {
              const fd = new FormData();
              fd.append("audio_file", f.file);
              fd.append("summarization_model", summarizationModel);
              const res = await callsApi.uploadCall(fd);
              callId = res.data?.call_id;
              if (!callId) throw new Error("No call ID returned from upload");
              addStoredCall(callId, f.name, f.size);
            }
            await pollCallStatus(callId, i);
          } catch (err: any) {
            failFile(i, err.response?.data?.error ?? err.message ?? "Analysis failed");
          }
        }),
      );

      setFileStates((prev) => {
        handleAllDone(prev);
        return prev;
      });
    };

    run();
    return () => {
      stepTimersRef.current.forEach(clearTimeout);
      pollTimersRef.current.forEach(clearInterval);
    };
  }, [
    files,
    fileStates.length,
    isRecoveryMode,
    summarizationModel,
    advanceSteps,
    pollCallStatus,
    failFile,
    handleAllDone,
  ]);

  // Process files — recovery mode
  useEffect(() => {
    if (!isRecoveryMode || fileStates.length === 0 || processingRef.current)
      return;
    processingRef.current = true;

    const active = getActiveAnalysis();
    if (!active?.calls.length) return;

    const run = async () => {
      await Promise.all(
        active.calls.map(async (c, i) => {
          advanceSteps(i, c.currentStepIndex ?? 2, c.fileName);
          await pollCallStatus(c.callId, i);
        }),
      );

      setFileStates((prev) => {
        handleAllDone(prev);
        return prev;
      });
    };

    run();
    return () => {
      stepTimersRef.current.forEach(clearTimeout);
      pollTimersRef.current.forEach(clearInterval);
    };
  }, [isRecoveryMode, fileStates.length, advanceSteps, pollCallStatus, handleAllDone]);

  // Auto-open detail modal when a single file completes
  useEffect(() => {
    const done = fileStates.filter((f) => f.status === "completed" && f.callId);
    if (
      done.length === 1 &&
      fileStates.length === 1 &&
      viewingCallId === null &&
      !hasAutoOpenedRef.current
    ) {
      setViewingCallId(done[0].callId!);
      hasAutoOpenedRef.current = true;
    }
  }, [fileStates, viewingCallId]);

  const goBack = () => {
    clearFiles();
    clearActiveAnalysis();
    if (onDone) onDone();
    else router.push("/dashboard/upload-call");
  };

  const handleModalClose = () => {
    setViewingCallId(null);
    if (overallStatus === "completed") goBack();
  };

  if (files.length === 0 && !isRecoveryMode) return null;

  const completedCount = fileStates.filter((f) => f.status === "completed").length;
  const errorCount = fileStates.filter((f) => f.status === "error").length;
  const totalCount = fileStates.length;
  const allDone = totalCount > 0 && completedCount + errorCount === totalCount;

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
          {!allDone ? (
            <div
              className="w-12 h-12 rounded-full flex items-center justify-center"
              style={{ background: "var(--accent-bg)" }}
            >
              <Loader2
                className="w-6 h-6 animate-spin"
                style={{ color: "var(--accent)" }}
              />
            </div>
          ) : (
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
            <span className="text-sm font-medium" style={{ color: "var(--text-secondary)" }}>
              Overall Progress
            </span>
            <span className="text-sm font-medium" style={{ color: "var(--text-secondary)" }}>
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

      {/* Per-file progress cards */}
      <div className="space-y-4">
        {fileStates.map((fs, i) => (
          <FileProgressCard
            key={i}
            fileState={fs}
            onViewCall={(id) => setViewingCallId(id)}
          />
        ))}
      </div>

      {/* Actions when done */}
      {allDone && (
        <div
          className="rounded-lg p-6 flex items-center justify-between"
          style={{ background: "#ffffff", border: "1px solid var(--border)" }}
        >
          <button
            onClick={goBack}
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

      {viewingCallId && (
        <CallDetailModal callId={viewingCallId} onClose={handleModalClose} />
      )}
    </div>
  );
}
