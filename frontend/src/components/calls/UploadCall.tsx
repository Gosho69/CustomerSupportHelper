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
    (f) => f.status === "success"
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
      })
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
        })
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
      <div className="bg-gradient-to-r from-blue-600 to-cyan-600 rounded-2xl p-8 text-white">
        <h1 className="text-3xl font-bold mb-2">Upload Call Recording</h1>
        <p className="text-blue-100">
          Upload your call recordings for AI analysis and performance insights
        </p>
      </div>

      {/* Upload Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center mb-4">
            <FileAudio className="w-5 h-5 text-blue-400" />
          </div>
          <p className="text-gray-400 text-sm mb-1">Accepted Formats</p>
          <p className="text-white font-semibold">MP3, WAV, M4A, OGG</p>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="w-10 h-10 bg-cyan-500/20 rounded-lg flex items-center justify-center mb-4">
            <Clock className="w-5 h-5 text-cyan-400" />
          </div>
          <p className="text-gray-400 text-sm mb-1">Max File Size</p>
          <p className="text-white font-semibold">100 MB</p>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center mb-4">
            <CheckCircle className="w-5 h-5 text-green-400" />
          </div>
          <p className="text-gray-400 text-sm mb-1">Analysis Time</p>
          <p className="text-white font-semibold">~2-5 minutes</p>
        </div>
      </div>

      {/* Upload Area */}
      {uploadedFiles.length === 0 ? (
        <div
          className={`bg-slate-800/50 backdrop-blur-md border-2 border-dashed rounded-2xl p-12 transition-all ${
            isDragging
              ? "border-blue-500 bg-blue-500/10"
              : "border-white/10 hover:border-white/20"
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="flex flex-col items-center justify-center text-center">
            <div className="w-20 h-20 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full flex items-center justify-center mb-6">
              <Upload className="w-10 h-10 text-white" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-2">
              Drop your audio files here
            </h3>
            <p className="text-gray-400 mb-6 max-w-md">
              or click to browse from your computer. Supports MP3, WAV, M4A,
              OGG, and WebM formats
            </p>
            <label className="px-8 py-3 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white rounded-xl font-semibold cursor-pointer transition-all transform hover:scale-105">
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
        <div className="bg-slate-800/50 backdrop-blur-md border-2 border-dashed border-white/10 rounded-2xl p-8">
          {/* File List in Upload Box */}
          <div className="space-y-3 mb-6">
            {uploadedFiles.map((uploadedFile, index) => (
              <div
                key={index}
                className="bg-slate-900/50 rounded-xl p-4 border border-white/5"
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
                      <p className="text-white font-medium truncate">
                        {uploadedFile.file.name}
                      </p>
                      <div className="flex items-center space-x-3 text-sm text-gray-400">
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
                    className="w-8 h-8 bg-slate-800 hover:bg-red-500/20 rounded-lg flex items-center justify-center transition-colors"
                  >
                    <X className="w-4 h-4 text-gray-400 hover:text-red-400" />
                  </button>
                </div>

                {/* Progress Bar */}
                {uploadedFile.status === "uploading" && (
                  <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 transition-all duration-300"
                      style={{ width: `${uploadedFile.progress}%` }}
                    />
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Analysis Button */}
          {hasSuccessfulUploads && allFilesUploaded && (
            <div className="pt-4 border-t border-white/10">
              <button
                onClick={handleStartAnalysis}
                disabled={isAnalyzing}
                className="w-full px-6 py-4 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed text-white rounded-xl font-semibold transition-all transform hover:scale-105 disabled:transform-none flex items-center justify-center space-x-2"
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
        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="flex items-start space-x-3">
            <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
              <CheckCircle className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <h4 className="text-white font-semibold mb-2">Best Practices</h4>
              <ul className="text-gray-400 text-sm space-y-2">
                <li>• Ensure clear audio quality for accurate analysis</li>
                <li>• Upload calls within 24 hours for timely feedback</li>
                <li>• Include complete conversations (intro to outro)</li>
                <li>• Avoid background noise when possible</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="flex items-start space-x-3">
            <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
              <AlertCircle className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <h4 className="text-white font-semibold mb-2">What You'll Get</h4>
              <ul className="text-gray-400 text-sm space-y-2">
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
