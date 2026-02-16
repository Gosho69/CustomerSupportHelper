"use client";

import { useState, useCallback } from "react";
import {
  Upload,
  File,
  CheckCircle,
  AlertCircle,
  Loader2,
  X,
  Music,
  Clock,
  FileAudio,
} from "lucide-react";

interface UploadedFile {
  file: File;
  progress: number;
  status: "pending" | "uploading" | "success" | "error";
  error?: string;
}

export default function UploadCall() {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const hasSuccessfulUploads = uploadedFiles.some(
    (f) => f.status === "success",
  );
  const allFilesUploaded =
    uploadedFiles.length > 0 &&
    uploadedFiles.every((f) => f.status === "success" || f.status === "error");

  const handleStartAnalysis = () => {
    setIsAnalyzing(true);
    // Simulate analysis - in production, this would be an API call
    setTimeout(() => {
      alert("Analysis started! You will be notified when complete.");
      setUploadedFiles([]);
      setIsAnalyzing(false);
    }, 1500);
  };

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

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
        // Simulate upload
        simulateUpload(file);
      }
    });
  };

  const simulateUpload = (file: File) => {
    setUploadedFiles((prev) =>
      prev.map((f) => {
        if (f.file === file) {
          return { ...f, status: "uploading" };
        }
        return f;
      }),
    );

    const interval = setInterval(() => {
      setUploadedFiles((prev) =>
        prev.map((f) => {
          if (f.file === file && f.status === "uploading") {
            const newProgress = Math.min(f.progress + 10, 100);
            const newStatus = newProgress === 100 ? "success" : "uploading";

            if (newProgress === 100) {
              clearInterval(interval);
            }

            return {
              ...f,
              progress: newProgress,
              status: newStatus,
            };
          }
          return f;
        }),
      );
    }, 300);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    processFiles(files);
  }, []);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      processFiles(e.target.files);
    }
  };

  const removeFile = (file: File) => {
    setUploadedFiles((prev) => prev.filter((f) => f.file !== file));
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
  };

  const formatDuration = (file: File) => {
    // This would need actual audio file parsing in production
    return "Unknown";
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
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div
          className="rounded-lg p-6"
          style={{
            background: "#ffffff",
            border: "1px solid var(--border)",
          }}
        >
          <div
            className="w-10 h-10 rounded-lg flex items-center justify-center mb-4"
            style={{ background: "var(--accent-bg)" }}
          >
            <FileAudio className="w-5 h-5" style={{ color: "var(--accent)" }} />
          </div>
          <p
            className="text-sm mb-1"
            style={{ color: "var(--text-secondary)" }}
          >
            Accepted Formats
          </p>
          <p className="font-semibold" style={{ color: "var(--text-primary)" }}>
            MP3, WAV, M4A, OGG
          </p>
        </div>

        <div
          className="rounded-lg p-6"
          style={{
            background: "#ffffff",
            border: "1px solid var(--border)",
          }}
        >
          <div
            className="w-10 h-10 rounded-lg flex items-center justify-center mb-4"
            style={{ background: "var(--accent-bg)" }}
          >
            <Clock className="w-5 h-5" style={{ color: "var(--accent)" }} />
          </div>
          <p
            className="text-sm mb-1"
            style={{ color: "var(--text-secondary)" }}
          >
            Max File Size
          </p>
          <p className="font-semibold" style={{ color: "var(--text-primary)" }}>
            100 MB
          </p>
        </div>

        <div
          className="rounded-lg p-6"
          style={{
            background: "#ffffff",
            border: "1px solid var(--border)",
          }}
        >
          <div
            className="w-10 h-10 rounded-lg flex items-center justify-center mb-4"
            style={{ background: "var(--accent-bg)" }}
          >
            <CheckCircle
              className="w-5 h-5"
              style={{ color: "var(--accent)" }}
            />
          </div>
          <p
            className="text-sm mb-1"
            style={{ color: "var(--text-secondary)" }}
          >
            Analysis Time
          </p>
          <p className="font-semibold" style={{ color: "var(--text-primary)" }}>
            ~2-5 minutes
          </p>
        </div>
      </div>

      {/* Upload Area */}
      {uploadedFiles.length === 0 ? (
        <div
          className={`border-2 border-dashed rounded-lg p-12 transition-all ${
            isDragging ? "border-blue-500 bg-blue-50" : "hover:border-gray-300"
          }`}
          style={{
            background: isDragging ? undefined : "#ffffff",
            borderColor: isDragging ? undefined : "var(--border)",
          }}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="flex flex-col items-center justify-center text-center">
            <div
              className="w-20 h-20 rounded-full flex items-center justify-center mb-6"
              style={{ background: "var(--accent-bg)" }}
            >
              <Upload
                className="w-10 h-10"
                style={{ color: "var(--accent)" }}
              />
            </div>
            <h3
              className="text-2xl font-bold mb-2"
              style={{ color: "var(--text-primary)" }}
            >
              Drop your audio files here
            </h3>
            <p
              className="mb-6 max-w-md"
              style={{ color: "var(--text-secondary)" }}
            >
              or click to browse from your computer. Supports MP3, WAV, M4A,
              OGG, and WebM formats
            </p>
            <label
              className="px-8 py-3 rounded-lg font-semibold cursor-pointer transition-all transform hover:scale-105"
              style={{ background: "var(--accent)", color: "#ffffff" }}
            >
              <input
                type="file"
                className="hidden"
                accept="audio/*,.mp3,.wav,.m4a,.ogg,.webm"
                onChange={handleFileInput}
              />
              Select Files
            </label>
          </div>
        </div>
      ) : (
        <div
          className="border-2 border-dashed rounded-lg p-8"
          style={{ background: "#ffffff", borderColor: "var(--border)" }}
        >
          {/* File List in Upload Box */}
          <div className="space-y-3 mb-6">
            {uploadedFiles.map((uploadedFile, index) => (
              <div
                key={index}
                className="rounded-lg p-4"
                style={{
                  background: "var(--background)",
                  border: "1px solid var(--border)",
                }}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center space-x-3 flex-1">
                    <div
                      className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                        uploadedFile.status === "success"
                          ? "bg-green-500/20"
                          : uploadedFile.status === "error"
                            ? "bg-red-500/20"
                            : uploadedFile.status === "uploading"
                              ? "bg-blue-500/20"
                              : "bg-gray-500/20"
                      }`}
                    >
                      {uploadedFile.status === "success" ? (
                        <CheckCircle className="w-5 h-5 text-green-400" />
                      ) : uploadedFile.status === "error" ? (
                        <AlertCircle className="w-5 h-5 text-red-400" />
                      ) : uploadedFile.status === "uploading" ? (
                        <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
                      ) : (
                        <File className="w-5 h-5 text-gray-400" />
                      )}
                    </div>
                    <div className="flex-1">
                      <p
                        className="font-medium truncate"
                        style={{ color: "var(--text-primary)" }}
                      >
                        {uploadedFile.file.name}
                      </p>
                      <div
                        className="flex items-center space-x-3 text-sm"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        <span>{formatFileSize(uploadedFile.file.size)}</span>
                        {uploadedFile.status === "success" && (
                          <span className="text-green-400 flex items-center">
                            <CheckCircle className="w-3 h-3 mr-1" />
                            Ready
                          </span>
                        )}
                        {uploadedFile.status === "error" && (
                          <span className="text-red-400">
                            {uploadedFile.error}
                          </span>
                        )}
                        {uploadedFile.status === "uploading" && (
                          <span className="text-blue-400">
                            {uploadedFile.progress}%
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={() => removeFile(uploadedFile.file)}
                    className="w-8 h-8 hover:bg-red-50 rounded-lg flex items-center justify-center transition-colors"
                    style={{
                      background: "var(--background)",
                      border: "1px solid var(--border)",
                    }}
                  >
                    <X
                      className="w-4 h-4"
                      style={{ color: "var(--text-secondary)" }}
                    />
                  </button>
                </div>

                {/* Progress Bar */}
                {uploadedFile.status === "uploading" && (
                  <div className="w-full bg-gray-100 rounded-full h-2 overflow-hidden">
                    <div
                      className="h-full transition-all duration-300"
                      style={{
                        background:
                          "linear-gradient(90deg, rgba(237,231,216,1) 0%, rgba(216,205,191,1) 100%)",
                      }}
                      style={{ width: `${uploadedFile.progress}%` }}
                    />
                  </div>
                )}
              </div>
            ))}
          </div>

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
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div
          className="rounded-lg p-6"
          style={{ background: "#ffffff", border: "1px solid var(--border)" }}
        >
          <div className="flex items-start space-x-3">
            <div className="w-10 h-10 bg-blue-50 rounded-lg flex items-center justify-center flex-shrink-0">
              <CheckCircle
                className="w-5 h-5"
                style={{ color: "var(--accent)" }}
              />
            </div>
            <div>
              <h4
                className="font-semibold mb-2"
                style={{ color: "var(--text-primary)" }}
              >
                Best Practices
              </h4>
              <ul
                className="text-sm space-y-2"
                style={{ color: "var(--text-secondary)" }}
              >
                <li>• Ensure clear audio quality for accurate analysis</li>
                <li>• Upload calls within 24 hours for timely feedback</li>
                <li>• Include complete conversations (intro to outro)</li>
                <li>• Avoid background noise when possible</li>
              </ul>
            </div>
          </div>
        </div>

        <div
          className="rounded-lg p-6"
          style={{ background: "#ffffff", border: "1px solid var(--border)" }}
        >
          <div className="flex items-start space-x-3">
            <div
              className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0"
              style={{ background: "var(--accent-bg)" }}
            >
              <AlertCircle
                className="w-5 h-5"
                style={{ color: "var(--accent)" }}
              />
            </div>
            <div>
              <h4
                className="font-semibold mb-2"
                style={{ color: "var(--text-primary)" }}
              >
                What You'll Get
              </h4>
              <ul
                className="text-sm space-y-2"
                style={{ color: "var(--text-secondary)" }}
              >
                <li>• Emotional sentiment analysis</li>
                <li>• Behavioral pattern insights</li>
                <li>• Personalized coaching tips</li>
                <li>• Performance scoring</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
