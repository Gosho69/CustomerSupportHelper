"use client";

import { useState, useEffect } from "react";
import { Phone } from "lucide-react";
import { PageHeader, SearchInput, StatsCard, Badge, CSATBadge } from "@/components/ui";
import CallDetailModal from "@/components/call-detail";
import { callsApi } from "@/lib/api";

interface Call {
  id: number;
  agentName: string;
  callDate: string;
  duration: number;
  sentiment: "positive" | "neutral" | "negative";
  status: "analyzed" | "pending" | "failed";
  score?: number;
  predictedCsat?: number | null;
  predictedCsatLabel?: string | null;
}

export default function AllCalls() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCallId, setSelectedCallId] = useState<number | null>(null);
  const [calls, setCalls] = useState<Call[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchCalls = async () => {
      try {
        setLoading(true);
        const response = await callsApi.getMyCalls();
        const data = response.data || [];

        const mapped: Call[] = data.map((call: any) => ({
          id: call.id,
          agentName: call.agent_name || call.agent?.username || "Unknown",
          callDate: call.call_date
            ? new Date(call.call_date).toLocaleString()
            : "",
          duration: call.duration || 0,
          sentiment: call.sentiment || "neutral",
          status: call.status || "analyzed",
          score: call.score,
          predictedCsat: call.predicted_csat ?? null,
          predictedCsatLabel: call.predicted_csat_label ?? null,
        }));

        setCalls(mapped);
      } catch (error) {
        console.error("Failed to fetch calls:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchCalls();
  }, []);

  const filteredCalls = calls.filter((call) => {
    const matchesSearch = call.agentName
      .toLowerCase()
      .includes(searchQuery.toLowerCase());
    return matchesSearch;
  });

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div style={{ color: "var(--text-secondary)" }}>Loading calls...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="All Team Calls"
        subtitle="View and analyze call recordings from your team"
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatsCard
          icon={Phone}
          iconColor="bg-gray-50 text-gray-500"
          label="Total Calls"
          value={calls.length}
        />
        <StatsCard
          icon={Phone}
          iconColor="bg-gray-50 text-gray-500"
          label="Analyzed"
          value={calls.filter((c) => c.status === "analyzed").length}
        />
        <StatsCard
          icon={Phone}
          iconColor="bg-gray-50 text-gray-500"
          label="Avg Duration"
          value={
            calls.length > 0
              ? formatDuration(
                  Math.round(
                    calls.reduce((sum, c) => sum + c.duration, 0) /
                      calls.length,
                  ),
                )
              : "0:00"
          }
        />
      </div>

      <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
        <SearchInput
          value={searchQuery}
          onChange={setSearchQuery}
          placeholder="Search calls..."
          className="flex-1 w-full md:max-w-md"
        />
      </div>

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
          <Phone
            className="w-5 h-5 mr-2"
            style={{ color: "var(--text-secondary)" }}
          />
          Call Recordings ({filteredCalls.length})
        </h2>

        <div className="space-y-3">
          {filteredCalls.map((call) => (
            <div
              key={call.id}
              className="rounded-lg p-5 hover:bg-gray-50 transition-all cursor-pointer"
              style={{
                background: "var(--background)",
                border: "1px solid var(--border)",
              }}
              onClick={() => setSelectedCallId(call.id)}
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
                      {call.agentName.charAt(0)}
                    </span>
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-1">
                      <p
                        className="font-semibold text-lg"
                        style={{ color: "var(--text-primary)" }}
                      >
                        {call.agentName}
                      </p>
                      <Badge variant="gray" size="sm">
                        {call.sentiment}
                      </Badge>
                      <CSATBadge score={call.predictedCsat} label={call.predictedCsatLabel} />
                    </div>
                    <p
                      className="text-sm"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {call.callDate}
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
                        style={{ color: "var(--text-primary)" }}
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

      {selectedCallId && (
        <CallDetailModal
          callId={selectedCallId}
          onClose={() => setSelectedCallId(null)}
        />
      )}
    </div>
  );
}
