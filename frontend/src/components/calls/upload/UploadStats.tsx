import React from "react";
import { FileAudio, Clock, CheckCircle } from "lucide-react";

export default function UploadStats() {
  const stats = [
    {
      icon: (
        <FileAudio className="w-5 h-5" style={{ color: "var(--accent)" }} />
      ),
      label: "Accepted Formats",
      value: "MP3, WAV, M4A, OGG",
    },
    {
      icon: <Clock className="w-5 h-5" style={{ color: "var(--accent)" }} />,
      label: "Max File Size",
      value: "100 MB",
    },
    {
      icon: (
        <CheckCircle className="w-5 h-5" style={{ color: "var(--accent)" }} />
      ),
      label: "Analysis Time",
      value: "~2-5 minutes",
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {stats.map((stat) => (
        <div
          key={stat.label}
          className="rounded-lg p-6"
          style={{ background: "#ffffff", border: "1px solid var(--border)" }}
        >
          <div
            className="w-10 h-10 rounded-lg flex items-center justify-center mb-4"
            style={{ background: "var(--accent-bg)" }}
          >
            {stat.icon}
          </div>
          <p
            className="text-sm mb-1"
            style={{ color: "var(--text-secondary)" }}
          >
            {stat.label}
          </p>
          <p className="font-semibold" style={{ color: "var(--text-primary)" }}>
            {stat.value}
          </p>
        </div>
      ))}
    </div>
  );
}
