import { useState } from "react";
import {
  Phone,
  Mail,
  Calendar,
  TrendingUp,
  TrendingDown,
  Eye,
} from "lucide-react";
import { Modal, StatusIndicator } from "@/components/ui";
import { PerformanceAreaChart } from "@/components/charts";
import CallDetailModal from "@/components/call-detail";

interface CallItem {
  id: number;
  agent: number;
  agent_name: string;
  call_date: string;
  duration: number | null;
  behavioral_score: number | null;
  created_at: string;
}

interface TeamMember {
  id: number;
  name: string;
  email: string;
  phone: string;
  joinDate: string;
  totalCalls: number;
  avgScore: number;
  avgCallDuration: string;
  trend: "up" | "down" | "stable";
  status: "active" | "on_break" | "offline";
  performanceData: { month: string; score: number }[];
}

interface MemberDetailModalProps {
  member: TeamMember | null;
  agentCalls?: CallItem[];
  onClose: () => void;
}

function formatDuration(seconds: number | null | undefined): string {
  if (!seconds || seconds <= 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function MemberDetailModal({
  member,
  agentCalls = [],
  onClose,
}: MemberDetailModalProps) {
  const [viewingCallId, setViewingCallId] = useState<number | null>(null);

  if (!member) return null;

  const getTrendIcon = (trend: "up" | "down" | "stable") => {
    switch (trend) {
      case "up":
        return (
          <TrendingUp
            className="w-4 h-4"
            style={{ color: "var(--success, #0caf60)" }}
          />
        );
      case "down":
        return (
          <TrendingDown
            className="w-4 h-4"
            style={{ color: "var(--error, #e53935)" }}
          />
        );
      default:
        return <div className="w-4 h-4 bg-gray-400 rounded-full" />;
    }
  };

  return (
    <>
      <Modal isOpen={!!member} onClose={onClose} size="4xl">
        {/* Header */}
        <div className="flex items-start space-x-6 mb-8">
          <div
            className="w-20 h-20 rounded-full flex items-center justify-center"
            style={{ background: "var(--accent-bg)" }}
          >
            <span
              className="font-bold text-3xl"
              style={{ color: "var(--accent)" }}
            >
              {member.name.charAt(0)}
            </span>
          </div>
          <div className="flex-1">
            <div className="flex items-center space-x-3 mb-2">
              <h2
                className="text-2xl font-bold"
                style={{ color: "var(--text-primary)" }}
              >
                {member.name}
              </h2>
              <StatusIndicator status={member.status} />
            </div>
            <div
              className="space-y-1"
              style={{ color: "var(--text-secondary)" }}
            >
              <p className="flex items-center">
                <Mail className="w-4 h-4 mr-2" />
                {member.email}
              </p>
              <p className="flex items-center">
                <Phone className="w-4 h-4 mr-2" />
                {member.phone || "—"}
              </p>
              {member.joinDate && (
                <p className="flex items-center">
                  <Calendar className="w-4 h-4 mr-2" />
                  Joined {member.joinDate}
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-4 gap-4 mb-8">
          {[
            { label: "Total Calls", value: member.totalCalls },
            { label: "Avg Score", value: `${member.avgScore}%` },
            { label: "Avg Duration", value: member.avgCallDuration },
          ].map(({ label, value }) => (
            <div
              key={label}
              className="rounded-lg p-4"
              style={{
                background: "#ffffff",
                border: "1px solid var(--border)",
              }}
            >
              <p
                className="text-sm mb-1"
                style={{ color: "var(--text-secondary)" }}
              >
                {label}
              </p>
              <p
                className="text-2xl font-bold"
                style={{ color: "var(--text-primary)" }}
              >
                {value}
              </p>
            </div>
          ))}
          <div
            className="rounded-lg p-4"
            style={{ background: "#ffffff", border: "1px solid var(--border)" }}
          >
            <p
              className="text-sm mb-1"
              style={{ color: "var(--text-secondary)" }}
            >
              Trend
            </p>
            <div className="flex items-center space-x-2 mt-2">
              {getTrendIcon(member.trend)}
              <span
                className="font-semibold capitalize"
                style={{ color: "var(--text-primary)" }}
              >
                {member.trend}
              </span>
            </div>
          </div>
        </div>

        {/* Performance History */}
        <div
          className="rounded-lg p-6 mb-6"
          style={{
            background: "var(--background)",
            border: "1px solid var(--border)",
          }}
        >
          <h3
            className="text-lg font-semibold mb-4"
            style={{ color: "var(--text-primary)" }}
          >
            Performance History
          </h3>
          <PerformanceAreaChart data={member.performanceData} />
        </div>

        {/* Agent's Calls */}
        <div
          className="rounded-lg p-6"
          style={{ background: "#ffffff", border: "1px solid var(--border)" }}
        >
          <h3
            className="text-lg font-semibold mb-4 flex items-center"
            style={{ color: "var(--text-primary)" }}
          >
            <Phone
              className="w-5 h-5 mr-2"
              style={{ color: "var(--accent)" }}
            />
            Call Recordings ({agentCalls.length})
          </h3>

          {agentCalls.length === 0 ? (
            <p
              className="text-sm py-4 text-center"
              style={{ color: "var(--text-secondary)" }}
            >
              No calls recorded yet.
            </p>
          ) : (
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {agentCalls.map((call) => (
                <div
                  key={call.id}
                  className="flex items-center justify-between rounded-lg px-4 py-3"
                  style={{
                    background: "var(--background)",
                    border: "1px solid var(--border)",
                  }}
                >
                  <div className="flex items-center space-x-4">
                    <div
                      className="w-8 h-8 rounded-full flex items-center justify-center"
                      style={{ background: "var(--accent-bg)" }}
                    >
                      <Phone
                        className="w-4 h-4"
                        style={{ color: "var(--accent)" }}
                      />
                    </div>
                    <div>
                      <p
                        className="text-sm font-medium"
                        style={{ color: "var(--text-primary)" }}
                      >
                        {formatDate(call.call_date)}
                      </p>
                      <p
                        className="text-xs"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        Duration: {formatDuration(call.duration)}
                        {call.behavioral_score != null &&
                        call.behavioral_score > 0
                          ? ` · Score: ${call.behavioral_score}%`
                          : ""}
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() => setViewingCallId(call.id)}
                    className="flex items-center space-x-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors hover:opacity-80"
                    style={{
                      background: "var(--accent-bg)",
                      color: "var(--accent)",
                    }}
                  >
                    <Eye className="w-3.5 h-3.5" />
                    <span>View Analysis</span>
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </Modal>

      {/* Call Detail Modal - outside main modal to avoid z-index stacking */}
      {viewingCallId && (
        <CallDetailModal
          callId={viewingCallId}
          onClose={() => setViewingCallId(null)}
        />
      )}
    </>
  );
}
