import React from "react";
import { Upload } from "lucide-react";

interface DropZoneProps {
  isDragging: boolean;
  onDragOver: (e: React.DragEvent) => void;
  onDragLeave: (e: React.DragEvent) => void;
  onDrop: (e: React.DragEvent) => void;
  onFileInput: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

export default function DropZone({
  isDragging,
  onDragOver,
  onDragLeave,
  onDrop,
  onFileInput,
}: DropZoneProps) {
  return (
    <div
      className={`border-2 border-dashed rounded-lg p-12 transition-all ${
        isDragging ? "border-blue-500 bg-blue-50" : "hover:border-gray-300"
      }`}
      style={{
        background: isDragging ? undefined : "#ffffff",
        borderColor: isDragging ? undefined : "var(--border)",
      }}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <div className="flex flex-col items-center justify-center text-center">
        <div
          className="w-20 h-20 rounded-full flex items-center justify-center mb-6"
          style={{ background: "var(--accent-bg)" }}
        >
          <Upload className="w-10 h-10" style={{ color: "var(--accent)" }} />
        </div>
        <h3
          className="text-2xl font-bold mb-2"
          style={{ color: "var(--text-primary)" }}
        >
          Drop your audio files here
        </h3>
        <p className="mb-6 max-w-md" style={{ color: "var(--text-secondary)" }}>
          or click to browse from your computer. Supports MP3, WAV, M4A, OGG,
          and WebM formats
        </p>
        <label
          className="px-8 py-3 rounded-lg font-semibold cursor-pointer transition-all transform hover:scale-105"
          style={{ background: "var(--accent)", color: "#ffffff" }}
        >
          <input
            type="file"
            className="hidden"
            accept="audio/*,.mp3,.wav,.m4a,.ogg,.webm"
            onChange={onFileInput}
          />
          Select Files
        </label>
      </div>
    </div>
  );
}
