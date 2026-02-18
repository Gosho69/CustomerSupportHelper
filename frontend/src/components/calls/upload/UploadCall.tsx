"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { CheckCircle, Loader2 } from "lucide-react";
import { useToast } from "@/components/ui";
import { useAnalysisStore } from "@/store/analysisStore";
import DropZone from "./DropZone";
import FileListItem, { UploadedFile } from "./FileListItem";
import ModelSelector from "./ModelSelector";
import UploadStats from "./UploadStats";
import UploadTips from "./UploadTips";

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
};

const validateFile = (file: File): string | null => {
  const allowedTypes = [
    "audio/mpeg",
    "audio/wav",
    "audio/mp3",
    "audio/m4a",
    "audio/ogg",
    "audio/webm",
  ];
  const maxSize = 100 * 1024 * 1024; // 100MB

  if (
    !allowedTypes.includes(file.type) &&
    !file.name.match(/\.(mp3|wav|m4a|ogg|webm)$/i)
  ) {
    return "Invalid file type. Please upload audio files only (MP3, WAV, M4A, OGG, WebM)";
  }

  if (file.size > maxSize) {
    return "File size exceeds 100MB limit";
  }

  return null;
};

export default function UploadCall() {
  const router = useRouter();
  const toast = useToast();
  const { setFiles, setSummarizationModel } = useAnalysisStore();
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedModel, setSelectedModel] = useState<"gpt4" | "local">("gpt4");

  const hasSuccessfulUploads = uploadedFiles.some(
    (f) => f.status === "success",
  );
  const allFilesUploaded =
    uploadedFiles.length > 0 &&
    uploadedFiles.every((f) => f.status === "success" || f.status === "error");

  const handleStartAnalysis = () => {
    const successfulFiles = uploadedFiles.filter((f) => f.status === "success");

    if (successfulFiles.length === 0) {
      toast.error("No files ready for analysis.");
      return;
    }

    setFiles(
      successfulFiles.map((f) => ({
        file: f.file,
        name: f.file.name,
        size: f.file.size,
      })),
    );
    setSummarizationModel(selectedModel);

    setUploadedFiles([]);
    router.push("/dashboard/upload-call/analysis");
  };

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const processFiles = (files: FileList | File[]) => {
    const fileArray = Array.from(files);

    fileArray.forEach((file) => {
      const error = validateFile(file);

      const newFile: UploadedFile = {
        file,
        progress: 0,
        status: error ? "error" : "pending",
        error: error || undefined,
      };

      setUploadedFiles((prev) => [...prev, newFile]);

      if (!error) {
        setUploadedFiles((prev) =>
          prev.map((f) =>
            f.file === file ? { ...f, status: "uploading" } : f,
          ),
        );

        let progress = 0;
        const interval = setInterval(() => {
          progress = Math.min(progress + 20, 100);
          setUploadedFiles((prev) =>
            prev.map((f) => {
              if (f.file === file && f.status === "uploading") {
                const newStatus = progress === 100 ? "success" : "uploading";
                if (progress === 100) clearInterval(interval);
                return { ...f, progress, status: newStatus };
              }
              return f;
            }),
          );
        }, 200);
      }
    });
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    processFiles(e.dataTransfer.files);
  }, []);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      processFiles(e.target.files);
    }
  };

  const removeFile = (file: File) => {
    setUploadedFiles((prev) => prev.filter((f) => f.file !== file));
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div
        className="rounded-lg p-8"
        style={{ background: "#ffffff", border: "1px solid var(--border)" }}
      >
        <h1
          className="text-3xl font-bold mb-2"
          style={{ color: "var(--text-primary)" }}
        >
          Upload Call Recording
        </h1>
        <p style={{ color: "var(--text-secondary)" }}>
          Upload your call recordings for AI analysis and performance insights
        </p>
      </div>

      {/* Upload Stats */}
      <UploadStats />

      {/* Upload Area */}
      {uploadedFiles.length === 0 ? (
        <DropZone
          isDragging={isDragging}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onFileInput={handleFileInput}
        />
      ) : (
        <div
          className="border-2 border-dashed rounded-lg p-8"
          style={{ background: "#ffffff", borderColor: "var(--border)" }}
        >
          {/* File List */}
          <div className="space-y-3 mb-6">
            {uploadedFiles.map((uploadedFile, index) => (
              <FileListItem
                key={index}
                uploadedFile={uploadedFile}
                onRemove={removeFile}
                formatFileSize={formatFileSize}
              />
            ))}
          </div>

          {/* Model Selector */}
          {hasSuccessfulUploads && allFilesUploaded && (
            <ModelSelector
              selectedModel={selectedModel}
              onSelectModel={setSelectedModel}
            />
          )}

          {/* Analysis Button */}
          {hasSuccessfulUploads && allFilesUploaded && (
            <div
              className="pt-4"
              style={{ borderTop: "1px solid var(--border)" }}
            >
              <button
                onClick={handleStartAnalysis}
                disabled={isAnalyzing}
                className="w-full px-6 py-4 rounded-lg font-semibold transition-all transform hover:scale-105 disabled:transform-none flex items-center justify-center space-x-2"
                style={{ background: "var(--accent)", color: "#ffffff" }}
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Starting Analysis...</span>
                  </>
                ) : (
                  <>
                    <CheckCircle className="w-5 h-5" />
                    <span>Send for Analysis</span>
                  </>
                )}
              </button>
            </div>
          )}
        </div>
      )}

      {/* Tips Section */}
      <UploadTips />
    </div>
  );
}
