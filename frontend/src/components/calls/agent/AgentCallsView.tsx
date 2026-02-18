"use client";

import { useEffect, useState } from "react";
import CallsHeader from "./CallsHeader";
import CallsStats from "./CallsStats";
import CallsFilters from "./CallsFilters";
import CallsTable, { Call } from "./CallsTable";
import CallDetailModal from "@/components/call-detail";
import { callsApi } from "@/lib/api";

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
      const response = await callsApi.getMyCalls();
      setCalls(response.data || []);
    } catch (error) {
      console.error("Failed to fetch calls:", error);
      setCalls([]);
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
            calls.reduce((sum, call) => sum + call.duration, 0) / calls.length,
          )
        : 0,
    ),
    today: calls.filter(
      (call) =>
        new Date(call.call_date).toDateString() === new Date().toDateString(),
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
