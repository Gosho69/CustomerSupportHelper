import { Phone } from "lucide-react";
import { Badge } from "@/components/ui";

export interface Call {
  id: number;
  agent_name: string;
  call_date: string;
  duration: number;
  created_at: string;
  sentiment?: "positive" | "neutral" | "negative";
  score?: number;
}

interface CallsTableProps {
  calls: Call[];
  loading: boolean;
  onViewDetails: (callId: number) => void;
}

export default function CallsTable({
  calls,
  loading,
  onViewDetails,
}: CallsTableProps) {
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  if (loading) {
    return (
      <div
        className="rounded-lg p-12"
        style={{
          background: "#ffffff",
          border: "1px solid var(--border)",
          borderRadius: "8px",
        }}
      >
        <div className="flex items-center justify-center">
          <div
            className="animate-spin rounded-full h-8 w-8 border-b-2"
            style={{ borderColor: "var(--accent)" }}
          ></div>
        </div>
      </div>
    );
  }

  return (
    <div
      className="rounded-lg p-6"
      style={{
        background: "#ffffff",
        border: "1px solid var(--border)",
        borderRadius: "8px",
      }}
    >
      <h2
        className="text-xl font-bold mb-6 flex items-center"
        style={{ color: "var(--text-primary)" }}
      >
        <Phone className="w-5 h-5 mr-2" style={{ color: "var(--accent)" }} />
        Call Recordings ({calls.length})
      </h2>

      <div className="space-y-3">
        {calls.map((call) => (
          <div
            key={call.id}
            className="rounded-lg p-5 hover:bg-gray-50 transition-all cursor-pointer"
            style={{
              background: "var(--background)",
              border: "1px solid var(--border)",
            }}
            onClick={() => onViewDetails(call.id)}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4 flex-1">
                <div
                  className="w-12 h-12 rounded-full flex items-center justify-center"
                  style={{ background: "var(--accent-bg)" }}
                >
                  <span
                    className="font-bold text-lg"
                    style={{ color: "var(--accent)" }}
                  >
                    {call.agent_name.charAt(0)}
                  </span>
                </div>
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-1">
                    <p
                      className="font-semibold text-lg"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {call.agent_name}
                    </p>
                    {call.sentiment && (
                      <Badge
                        variant={
                          call.sentiment === "positive"
                            ? "green"
                            : call.sentiment === "neutral"
                              ? "blue"
                              : "red"
                        }
                        size="sm"
                      >
                        {call.sentiment}
                      </Badge>
                    )}
                  </div>
                  <p
                    className="text-sm"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {formatDate(call.call_date)}
                  </p>
                </div>
              </div>

              <div className="flex items-center space-x-6">
                <div className="text-center">
                  <p
                    className="text-xs mb-1"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Duration
                  </p>
                  <p
                    className="font-semibold"
                    style={{ color: "var(--text-primary)" }}
                  >
                    {formatDuration(call.duration)}
                  </p>
                </div>
                {call.score && (
                  <div className="text-center">
                    <p
                      className="text-xs mb-1"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      Score
                    </p>
                    <p
                      className="font-bold text-lg"
                      style={{ color: "#0caf60" }}
                    >
                      {call.score}%
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
