"use client";

import { MessageSquare } from "lucide-react";
import { CallDetail, formatDuration } from "./callDetailHelpers";

interface TranscriptTabProps {
  call: CallDetail;
}

interface Utterance {
  role: string;
  start: number;
  end: number;
  text: string;
}

/** Merge consecutive utterances from the same role (frontend safety-net) */
function mergeConsecutiveSpeaker(utterances: Utterance[]): Utterance[] {
  const result: Utterance[] = [];
  for (const utt of utterances) {
    const prev = result[result.length - 1];
    if (prev && prev.role === utt.role) {
      prev.end = Math.max(prev.end, utt.end);
      prev.text = prev.text.trimEnd() + " " + utt.text.trimStart();
    } else {
      result.push({ ...utt });
    }
  }
  return result;
}

export default function TranscriptTab({ call }: TranscriptTabProps) {
  const rawUtterances: Utterance[] = call.transcript?.utterances || [];
  const fullText = call.transcript?.text || "";

  const utterances = mergeConsecutiveSpeaker(rawUtterances);

  // Collect all distinct non-agent roles for colour assignment
  const customerRoles = Array.from(
    new Set(
      utterances.map((u) => u.role).filter((r) => r.toLowerCase() !== "agent"),
    ),
  );

  const customerColors = [
    {
      bg: "bg-emerald-50",
      text: "text-emerald-700",
      border: "border-emerald-200",
      bubble: "#f0fdf4",
      label: "#065f46",
    },
    {
      bg: "bg-violet-50",
      text: "text-violet-700",
      border: "border-violet-200",
      bubble: "#f5f3ff",
      label: "#5b21b6",
    },
    {
      bg: "bg-amber-50",
      text: "text-amber-700",
      border: "border-amber-200",
      bubble: "#fffbeb",
      label: "#92400e",
    },
  ];

  const roleColorMap: Record<string, (typeof customerColors)[0]> = {};
  customerRoles.forEach((r, i) => {
    roleColorMap[r] = customerColors[i % customerColors.length];
  });

  return (
    <div
      className="rounded-lg p-6"
      style={{
        background: "var(--background)",
        border: "1px solid var(--border)",
      }}
    >
      <h3
        className="text-lg font-semibold mb-6 flex items-center"
        style={{ color: "var(--text-primary)" }}
      >
        <MessageSquare
          className="w-5 h-5 mr-2"
          style={{ color: "var(--accent)" }}
        />
        Full Transcript
      </h3>

      <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2">
        {utterances.length > 0 ? (
          utterances.map((utt, index) => {
            const role = utt.role || "Unknown";
            const isAgent = role.toLowerCase() === "agent";
            const customerColor = !isAgent ? roleColorMap[role] : null;

            return (
              <div
                key={index}
                className={`flex flex-col gap-1 ${isAgent ? "items-start" : "items-end"}`}
              >
                {/* Role label */}
                <span
                  className={`text-[11px] font-semibold uppercase tracking-wide px-2 ${
                    isAgent
                      ? "text-blue-500"
                      : (customerColor?.text ?? "text-emerald-700")
                  }`}
                >
                  {role}
                </span>

                {/* Bubble */}
                <div
                  className={`max-w-[78%] rounded-2xl px-4 py-3 shadow-sm border ${
                    isAgent
                      ? "rounded-tl-sm border-blue-100"
                      : `rounded-tr-sm ${customerColor?.border ?? "border-emerald-200"}`
                  }`}
                  style={{
                    background: isAgent
                      ? "#eff6ff"
                      : (customerColor?.bubble ?? "#f0fdf4"),
                  }}
                >
                  <p
                    className="text-sm leading-relaxed"
                    style={{ color: "var(--text-primary)" }}
                  >
                    {utt.text}
                  </p>
                  {utt.start !== undefined && (
                    <p
                      className={`text-[11px] mt-1.5 ${isAgent ? "text-blue-400" : (customerColor?.text ?? "text-emerald-600") + " opacity-70"}`}
                    >
                      {formatDuration(Math.round(utt.start))} –{" "}
                      {formatDuration(Math.round(utt.end))}
                    </p>
                  )}
                </div>
              </div>
            );
          })
        ) : fullText ? (
          <div
            className="whitespace-pre-wrap leading-relaxed text-sm"
            style={{ color: "var(--text-secondary)" }}
          >
            {fullText}
          </div>
        ) : (
          <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
            No transcript available
          </p>
        )}
      </div>
    </div>
  );
}
