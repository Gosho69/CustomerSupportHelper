"use client";

import { MessageSquare } from "lucide-react";
import { CallDetail, formatDuration } from "./callDetailHelpers";

interface TranscriptTabProps {
  call: CallDetail;
}

export default function TranscriptTab({ call }: TranscriptTabProps) {
  const utterances = call.transcript?.utterances || [];
  const fullText = call.transcript?.text || "";

  return (
    <div
      className="rounded-lg p-6"
      style={{
        background: "var(--background)",
        border: "1px solid var(--border)",
      }}
    >
      <h3
        className="text-lg font-semibold mb-4 flex items-center"
        style={{ color: "var(--text-primary)" }}
      >
        <MessageSquare
          className="w-5 h-5 mr-2"
          style={{ color: "var(--accent)" }}
        />
        Full Transcript
      </h3>
      <div className="space-y-3 max-h-[600px] overflow-y-auto pr-2">
        {utterances.length > 0 ? (
          utterances.map((utt: any, index: number) => (
            <div
              key={index}
              className="flex items-start space-x-3 p-3 rounded-lg"
              style={{ background: "var(--accent-bg)" }}
            >
              <div
                className={`px-3 py-1 rounded-full text-xs font-medium flex-shrink-0 ${
                  utt.speaker?.toLowerCase().includes("agent") ||
                  utt.speaker === "SPEAKER_01"
                    ? "bg-blue-50 text-blue-600"
                    : "bg-green-50 text-green-600"
                }`}
              >
                {(utt.speaker || "Unknown").toUpperCase()}
              </div>
              <div className="flex-1">
                <p style={{ color: "var(--text-secondary)" }}>{utt.text}</p>
                {utt.start !== undefined && (
                  <p
                    className="text-xs mt-1"
                    style={{ color: "var(--text-tertiary)" }}
                  >
                    {formatDuration(Math.round(utt.start))} –{" "}
                    {formatDuration(Math.round(utt.end))}
                  </p>
                )}
              </div>
            </div>
          ))
        ) : fullText ? (
          <div
            className="whitespace-pre-wrap leading-relaxed"
            style={{ color: "var(--text-secondary)" }}
          >
            {fullText}
          </div>
        ) : (
          <p style={{ color: "var(--text-secondary)" }}>
            No transcript available
          </p>
        )}
      </div>
    </div>
  );
}
