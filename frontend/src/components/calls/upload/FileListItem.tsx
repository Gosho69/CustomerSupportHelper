import React from "react";
import { File, CheckCircle, AlertCircle, Loader2, X } from "lucide-react";

export interface UploadedFile {
  file: File;
  progress: number;
  status: "pending" | "uploading" | "success" | "error";
  error?: string;
}

interface FileListItemProps {
  uploadedFile: UploadedFile;
  onRemove: (file: File) => void;
  formatFileSize: (bytes: number) => string;
}

export default function FileListItem({
  uploadedFile,
  onRemove,
  formatFileSize,
}: FileListItemProps) {
  return (
    <div
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
                <span className="text-red-400">{uploadedFile.error}</span>
              )}
              {uploadedFile.status === "uploading" && (
                <span className="text-blue-400">{uploadedFile.progress}%</span>
              )}
            </div>
          </div>
        </div>
        <button
          onClick={() => onRemove(uploadedFile.file)}
          className="w-8 h-8 hover:bg-red-50 rounded-lg flex items-center justify-center transition-colors"
          style={{
            background: "var(--background)",
            border: "1px solid var(--border)",
          }}
        >
          <X className="w-4 h-4" style={{ color: "var(--text-secondary)" }} />
        </button>
      </div>

      {/* Progress Bar */}
      {uploadedFile.status === "uploading" && (
        <div className="w-full bg-gray-100 rounded-full h-2 overflow-hidden">
          <div
            className="h-full transition-all duration-300"
            style={{
              width: `${uploadedFile.progress}%`,
              background: "var(--accent)",
            }}
          />
        </div>
      )}
    </div>
  );
}
