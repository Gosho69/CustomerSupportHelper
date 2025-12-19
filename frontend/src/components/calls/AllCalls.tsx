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
          iconColor="bg-blue-500/20 text-blue-400"
          label="Total Calls"
          value={calls.length}
        />
        <StatsCard
          icon={Phone}
          iconColor="bg-green-500/20 text-green-400"
          label="Analyzed"
          value={calls.filter((c) => c.status === "analyzed").length}
        />
        <StatsCard
          icon={Phone}
          iconColor="bg-yellow-500/20 text-yellow-400"
          label="Avg Duration"
          value={formatDuration(
            Math.round(
              calls.reduce((sum, c) => sum + c.duration, 0) / calls.length
            )
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
          <select
            value={filterSentiment}
            onChange={(e) => setFilterSentiment(e.target.value)}
            className="px-4 py-3 bg-slate-800/50 border border-white/10 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
          >
            <option value="all">All Sentiments</option>
            <option value="positive">Positive</option>
            <option value="neutral">Neutral</option>
            <option value="negative">Negative</option>
          </select>

          <button className="px-6 py-3 bg-slate-700/50 hover:bg-slate-700 text-white rounded-xl font-semibold transition-all flex items-center space-x-2">
            <Download className="w-5 h-5" />
            <span>Export</span>
          </button>
        </div>
      </div>

      <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
        <h2 className="text-xl font-bold text-white mb-6 flex items-center">
          <Phone className="w-5 h-5 mr-2 text-purple-400" />
          Call Recordings ({filteredCalls.length})
        </h2>

        <div className="space-y-3">
          {filteredCalls.map((call) => (
            <div
              key={call.id}
              className="bg-slate-900/50 rounded-xl p-5 hover:bg-slate-900/70 transition-all cursor-pointer"
              onClick={() => setSelectedCallId(call.id)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4 flex-1">
                  <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
                    <span className="text-white font-bold text-lg">
                      {call.agentName.charAt(0)}
                    </span>
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-1">
                      <p className="text-white font-semibold text-lg">
                        {call.agentName}
                      </p>
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
                    </div>
                    <p className="text-sm text-gray-400">{call.callDate}</p>
                  </div>
                </div>

                <div className="flex items-center space-x-6">
                  <div className="text-center">
                    <p className="text-gray-400 text-xs mb-1">Duration</p>
                    <p className="text-white font-semibold">
                      {formatDuration(call.duration)}
                    </p>
                  </div>
                  {call.score && (
                    <div className="text-center">
                      <p className="text-gray-400 text-xs mb-1">Score</p>
                      <p className="text-green-400 font-bold text-lg">
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
