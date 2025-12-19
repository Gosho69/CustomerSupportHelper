"use client";

import { useEffect, useState } from "react";
import CallsHeader from "./CallsHeader";
import CallsStats from "./CallsStats";
import CallsFilters from "./CallsFilters";
import CallsTable, { Call } from "./CallsTable";
import CallDetailModal from "@/components/CallDetailModal";

export default function AgentCallsView() {
  const [calls, setCalls] = useState<Call[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedCallId, setSelectedCallId] = useState<number | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [filterDuration, setFilterDuration] = useState<string>("all");

  useEffect(() => {
    fetchCalls();
  }, []);

  const fetchCalls = async () => {
    try {
      setLoading(true);
      // Mock data for demo - replace with: const response = await api.calls.list();

      const mockCalls: Call[] = [
        {
          id: 1,
          agent_name: "You",
          call_date: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
          duration: 456,
          created_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
          sentiment: "positive",
          score: 92,
        },
        {
          id: 2,
          agent_name: "You",
          call_date: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
          duration: 723,
          created_at: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
          sentiment: "neutral",
          score: 85,
        },
        {
          id: 3,
          agent_name: "You",
          call_date: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString(),
          duration: 312,
          created_at: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString(),
          sentiment: "negative",
          score: 68,
        },
        {
          id: 4,
          agent_name: "You",
          call_date: new Date(Date.now() - 72 * 60 * 60 * 1000).toISOString(),
          duration: 589,
          created_at: new Date(Date.now() - 72 * 60 * 60 * 1000).toISOString(),
          sentiment: "positive",
          score: 88,
        },
      ];
      setCalls(mockCalls);
    } catch (error) {
      console.error("Failed to fetch calls:", error);
    } finally {
      setLoading(false);
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const filteredCalls = calls.filter((call) => {
    // Search by date - match formatted date string
    const callDate = new Date(call.call_date).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
    const matchesSearch = callDate
      .toLowerCase()
      .includes(searchQuery.toLowerCase());

    const matchesDuration =
      filterDuration === "all" ||
      (filterDuration === "short" && call.duration < 300) ||
      (filterDuration === "medium" &&
        call.duration >= 300 &&
        call.duration < 600) ||
      (filterDuration === "long" && call.duration >= 600);
    return matchesSearch && matchesDuration;
  });

  const stats = {
    total: calls.length,
    avgDuration: formatDuration(
      calls.length > 0
        ? Math.round(
            calls.reduce((sum, call) => sum + call.duration, 0) / calls.length
          )
        : 0
    ),
    today: calls.filter(
      (call) =>
        new Date(call.call_date).toDateString() === new Date().toDateString()
    ).length,
  };

  return (
    <div className="space-y-6">
      <CallsHeader />

      <CallsStats
        total={stats.total}
        avgDuration={stats.avgDuration}
        today={stats.today}
      />

      <CallsFilters
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
        filterDuration={filterDuration}
        setFilterDuration={setFilterDuration}
      />

      <CallsTable
        calls={filteredCalls}
        loading={loading}
        onViewDetails={setSelectedCallId}
      />

      {selectedCallId && (
        <CallDetailModal
          callId={selectedCallId}
          onClose={() => setSelectedCallId(null)}
        />
      )}
    </div>
  );
}
