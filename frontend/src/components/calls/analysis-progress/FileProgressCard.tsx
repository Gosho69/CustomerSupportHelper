import React from "react";
import {
  FileAudio,
  CheckCircle,
  AlertCircle,
  Loader2,
  Eye,
} from "lucide-react";
import { FileAnalysisState, formatFileSize } from "./analysisConstants";
import { ANALYSIS_STEPS } from "./AnalysisSteps";

interface FileProgressCardProps {
  fileState: FileAnalysisState;
  onViewCall: (callId: number) => void;
}

export default function FileProgressCard({
  fileState,
  onViewCall,
}: FileProgressCardProps) {
  return (
    <div
      className="rounded-lg overflow-hidden"
      style={{
        background: "#ffffff",
        border: `1px solid ${
          fileState.status === "error"
            ? "#fca5a5"
            : fileState.status === "completed"
              ? "#86efac"
              : "var(--border)"
        }`,
      }}
    >
      {/* File header */}
      <div
        className="px-6 py-4 flex items-center justify-between"
        style={{
          borderBottom: "1px solid var(--border)",
          background:
            fileState.status === "completed"
              ? "#f0fdf4"
              : fileState.status === "error"
                ? "#fef2f2"
                : "var(--background)",
        }}
      >
        <div className="flex items-center space-x-3">
          <div
            className={`w-10 h-10 rounded-lg flex items-center justify-center ${
              fileState.status === "completed"
                ? "bg-green-100"
                : fileState.status === "error"
                  ? "bg-red-100"
                  : "bg-blue-50"
            }`}
          >
            {fileState.status === "completed" ? (
              <CheckCircle className="w-5 h-5 text-green-500" />
            ) : fileState.status === "error" ? (
              <AlertCircle className="w-5 h-5 text-red-500" />
            ) : (
              <FileAudio
                className="w-5 h-5"
                style={{ color: "var(--accent)" }}
              />
            )}
          </div>
          <div>
            <p
              className="font-semibold"
              style={{ color: "var(--text-primary)" }}
            >
              {fileState.fileName}
            </p>
            <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
              {formatFileSize(fileState.fileSize)}
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          {fileState.status === "processing" && (
            <span
              className="text-sm font-medium px-3 py-1 rounded-full bg-blue-50"
              style={{ color: "var(--accent)" }}
            >
              Processing...
            </span>
          )}
          {fileState.status === "completed" && (
            <div className="flex items-center space-x-2">
              <span className="text-sm font-medium px-3 py-1 rounded-full bg-green-100 text-green-700">
                Complete
              </span>
              {fileState.callId && (
                <button
                  onClick={() => onViewCall(fileState.callId!)}
                  className="flex items-center space-x-1 text-sm font-medium px-3 py-1 rounded-full transition-colors hover:opacity-80"
                  style={{
                    background: "var(--accent)",
                    color: "#ffffff",
                  }}
                >
                  <Eye className="w-3.5 h-3.5" />
                  <span>View</span>
                </button>
              )}
            </div>
          )}
          {fileState.status === "error" && (
            <span className="text-sm font-medium px-3 py-1 rounded-full bg-red-100 text-red-700">
              Failed
            </span>
          )}
        </div>
      </div>

      {/* Steps list */}
      <div className="px-6 py-4">
        {fileState.status === "error" && fileState.error && (
          <div className="mb-4 p-3 rounded-lg bg-red-50 border border-red-200">
            <p className="text-sm text-red-700">{fileState.error}</p>
          </div>
        )}

        <div className="relative">
          {/* Vertical line */}
          <div
            className="absolute left-[17px] top-3 bottom-3 w-0.5"
            style={{ background: "var(--border)" }}
          />

          <div className="space-y-1">
            {ANALYSIS_STEPS.map((step, stepIndex) => {
              let stepStatus: "pending" | "active" | "completed" | "error" =
                "pending";

              if (fileState.status === "error") {
                if (stepIndex < fileState.currentStepIndex)
                  stepStatus = "completed";
                else if (stepIndex === fileState.currentStepIndex)
                  stepStatus = "error";
              } else if (fileState.status === "completed") {
                stepStatus = "completed";
              } else {
                if (stepIndex < fileState.currentStepIndex)
                  stepStatus = "completed";
                else if (stepIndex === fileState.currentStepIndex)
                  stepStatus = "active";
              }

              return (
                <div
                  key={step.id}
                  className={`flex items-center space-x-4 py-2.5 px-2 rounded-lg transition-all duration-500 ${
                    stepStatus === "active" ? "bg-blue-50/50" : ""
                  }`}
                >
                  {/* Step icon */}
                  <div
                    className={`relative z-10 w-[34px] h-[34px] rounded-full flex items-center justify-center flex-shrink-0 transition-all duration-500 ${
                      stepStatus === "completed"
                        ? "bg-green-100 text-green-600"
                        : stepStatus === "active"
                          ? "text-white shadow-lg shadow-blue-200"
                          : stepStatus === "error"
                            ? "bg-red-100 text-red-500"
                            : "bg-gray-100 text-gray-400"
                    }`}
                    style={
                      stepStatus === "active"
                        ? { background: "var(--accent)" }
                        : undefined
                    }
                  >
                    {stepStatus === "completed" ? (
                      <CheckCircle className="w-4 h-4" />
                    ) : stepStatus === "active" ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : stepStatus === "error" ? (
                      <AlertCircle className="w-4 h-4" />
                    ) : (
                      <span className="w-2 h-2 rounded-full bg-gray-300" />
                    )}
                  </div>

                  {/* Step text */}
                  <div className="flex-1 min-w-0">
                    <p
                      className={`text-sm font-medium transition-colors duration-300 ${
                        stepStatus === "completed"
                          ? "text-green-700"
                          : stepStatus === "active"
                            ? ""
                            : stepStatus === "error"
                              ? "text-red-600"
                              : "text-gray-400"
                      }`}
                      style={
                        stepStatus === "active"
                          ? { color: "var(--text-primary)" }
                          : undefined
                      }
                    >
                      {step.label}
                    </p>
                    {stepStatus === "active" && (
                      <p
                        className="text-xs mt-0.5 animate-pulse"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        {step.description}
                      </p>
                    )}
                  </div>

                  {/* Step status indicator */}
                  {stepStatus === "active" && (
                    <div className="flex space-x-1">
                      <span
                        className="w-1.5 h-1.5 rounded-full animate-bounce"
                        style={{
                          background: "var(--accent)",
                          animationDelay: "0ms",
                        }}
                      />
                      <span
                        className="w-1.5 h-1.5 rounded-full animate-bounce"
                        style={{
                          background: "var(--accent)",
                          animationDelay: "150ms",
                        }}
                      />
                      <span
                        className="w-1.5 h-1.5 rounded-full animate-bounce"
                        style={{
                          background: "var(--accent)",
                          animationDelay: "300ms",
                        }}
                      />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
