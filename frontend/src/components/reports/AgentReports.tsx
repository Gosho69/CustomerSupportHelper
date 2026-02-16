"use client";

import { useState, useEffect } from "react";
import PerformanceHeader from "./agent/PerformanceHeader";
import PerformanceCharts from "./agent/PerformanceCharts";
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

  const performanceData = [
    { month: "Jul", score: 75, calls: 78 },
    { month: "Aug", score: 78, calls: 82 },
    { month: "Sep", score: 80, calls: 85 },
    { month: "Oct", score: 82, calls: 89 },
    { month: "Nov", score: 82, calls: 89 },
    { month: "Dec", score: 87, calls: 67 },
  ];

  const categoryScores = [
    { category: "Empathy", score: 92 },
    { category: "Communication", score: 88 },
    { category: "Problem Solving", score: 85 },
    { category: "Product Knowledge", score: 79 },
    { category: "Time Management", score: 83 },
  ];

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
    reports.reduce((acc, r) => acc + r.score, 0) / reports.length;
  const latestScore = reports[0].score;
  const scoreTrend = latestScore - reports[1].score;

  return (
    <div className="space-y-6">
      <PerformanceHeader
        latestScore={latestScore}
        avgScore={avgScore}
        totalReports={reports.length}
        scoreTrend={scoreTrend}
      />

      <PerformanceCharts
        performanceData={performanceData}
        categoryScores={categoryScores}
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
