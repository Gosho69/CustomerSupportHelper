"use client";

import { useState } from "react";
import { Phone, Search, Download } from "lucide-react";
import { PageHeader, SearchInput, StatsCard, Badge } from "@/components/ui";
import CallDetailModal from "@/components/CallDetailModal";

interface Call {
  id: number;
  agentName: string;
  callDate: string;
  duration: number;
  sentiment: "positive" | "neutral" | "negative";
  status: "analyzed" | "pending" | "failed";
  score?: number;
}

export default function AllCalls() {
  const [searchQuery, setSearchQuery] = useState("");
  const [filterSentiment, setFilterSentiment] = useState<string>("all");
  const [selectedCallId, setSelectedCallId] = useState<number | null>(null);

  const [calls] = useState<Call[]>([
    {
      id: 1,
      agentName: "John Smith",
      callDate: "Dec 17, 2024 10:30 AM",
      duration: 456,
      sentiment: "positive",
      status: "analyzed",
      score: 92,
    },
    {
      id: 2,
      agentName: "Sarah Johnson",
      callDate: "Dec 17, 2024 09:15 AM",
      duration: 723,
      sentiment: "neutral",
      status: "analyzed",
      score: 85,
    },
    {
      id: 3,
      agentName: "Mike Wilson",
      callDate: "Dec 16, 2024 04:20 PM",
      duration: 312,
      sentiment: "negative",
      status: "analyzed",
      score: 68,
    },
  ]);

  const filteredCalls = calls.filter((call) => {
    const matchesSearch = call.agentName
      .toLowerCase()
      .includes(searchQuery.toLowerCase());
    const matchesSentiment =
      filterSentiment === "all" || call.sentiment === filterSentiment;
    return matchesSearch && matchesSentiment;
  });

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

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
          value={formatDuration(
            Math.round(
              calls.reduce((sum, c) => sum + c.duration, 0) / calls.length,
            ),
          )}
        />
      </div>

      <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
        <SearchInput
          value={searchQuery}
          onChange={setSearchQuery}
          placeholder="Search calls..."
          className="flex-1 w-full md:max-w-md"
        />

        <div className="flex gap-3">
          <div className="relative">
            <select
              value={filterSentiment}
              onChange={(e) => setFilterSentiment(e.target.value)}
              className="px-4 py-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 appearance-none cursor-pointer pr-10"
              style={{
                background: "#ffffff",
                border: "1px solid var(--border)",
                color: "var(--text-primary)",
              }}
            >
              <option value="all">All Sentiments</option>
              <option value="positive">Positive</option>
              <option value="neutral">Neutral</option>
              <option value="negative">Negative</option>
            </select>
            <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
              <svg
                className="w-5 h-5"
                style={{ color: "var(--text-secondary)" }}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 9l-7 7-7-7"
                />
              </svg>
            </div>
          </div>

          <button
            className="px-6 py-3 hover:bg-gray-50 rounded-lg font-semibold transition-all flex items-center space-x-2"
            style={{
              background: "#ffffff",
              border: "1px solid var(--border)",
              color: "var(--text-primary)",
            }}
          >
            <Download className="w-5 h-5" />
            <span>Export</span>
          </button>
        </div>
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
