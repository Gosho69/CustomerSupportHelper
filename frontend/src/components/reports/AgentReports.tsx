"use client";

import { useState, useEffect } from "react";
import PerformanceHeader from "./agent/PerformanceHeader";
import ReportFilters from "./agent/ReportFilters";
import ReportListItem, { Report } from "./agent/ReportListItem";
import ReportDetailModal from "./agent/ReportDetailModal";
import EmptyState from "./agent/EmptyState";
import { reportsApi } from "@/lib/api";

export default function AgentReports() {
  const [filter, setFilter] = useState<"all" | "weekly" | "monthly">("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedReport, setSelectedReport] = useState<Report | null>(null);
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchReports = async () => {
      try {
        setLoading(true);
        const response = await reportsApi.getMyReports();
        setReports(response.data || []);
      } catch (error) {
        console.error("Failed to fetch reports:", error);
        setReports([]);
      } finally {
        setLoading(false);
      }
    };

    fetchReports();
  }, []);

  const filteredReports = reports.filter((report) => {
    if (filter !== "all" && report.type !== filter) return false;
    if (
      searchQuery &&
      !report.period.toLowerCase().includes(searchQuery.toLowerCase())
    ) {
      return false;
    }
    return true;
  });

  const avgScore =
    reports.length > 0
      ? Math.round(
          reports.reduce((acc, r) => acc + r.score, 0) / reports.length,
        )
      : 0;
  const latestScore = reports.length > 0 ? reports[0].score : 0;
  const scoreTrend = reports.length > 1 ? latestScore - reports[1].score : 0;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div style={{ color: "var(--text-secondary)" }}>Loading reports...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <PerformanceHeader
        latestScore={latestScore}
        avgScore={avgScore}
        totalReports={reports.length}
        scoreTrend={scoreTrend}
      />

      <ReportFilters
        filter={filter}
        setFilter={setFilter}
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
      />

      <div className="space-y-4">
        {filteredReports.map((report) => (
          <ReportListItem
            key={report.id}
            report={report}
            onViewDetails={setSelectedReport}
          />
        ))}
      </div>

      {selectedReport && (
        <ReportDetailModal
          report={selectedReport}
          onClose={() => setSelectedReport(null)}
        />
      )}

      {filteredReports.length === 0 && <EmptyState />}
    </div>
  );
}
