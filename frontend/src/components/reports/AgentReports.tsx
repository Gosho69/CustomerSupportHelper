"use client";

import { useState, useEffect } from "react";
import PerformanceHeader from "./agent/PerformanceHeader";
import ReportFilters from "./agent/ReportFilters";
import ReportListItem, { Report } from "./agent/ReportListItem";
import ReportDetailModal from "./agent/ReportDetailModal";
import EmptyState from "./agent/EmptyState";
import { reportsApi } from "@/lib/api";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function mapReport(r: any): Report {
  const fmt = (d: string) =>
    d
      ? new Date(d).toLocaleDateString("en-US", {
          month: "short",
          day: "numeric",
          year: "numeric",
        })
      : "";
  const start = fmt(r.start_date);
  const end = fmt(r.end_date);
  return {
    id: r.id,
    type: (r.report_type as "weekly" | "monthly") || "weekly",
    period: start && end ? `${start} – ${end}` : start || end,
    date: r.generated_at
      ? new Date(r.generated_at).toLocaleDateString("en-US", {
          year: "numeric",
          month: "short",
          day: "numeric",
        })
      : "",
    score: Math.round((r.average_behavioral_score ?? 0) * 100),
    trend:
      r.behavioral_trend === "improving"
        ? "up"
        : r.behavioral_trend === "declining"
          ? "down"
          : "stable",
    totalCalls: r.total_calls ?? 0,
    avgDuration: r.average_call_duration ?? 0,
    strengths: Array.isArray(r.strengths) ? r.strengths : [],
    improvements: Array.isArray(r.weaknesses)
      ? r.weaknesses
      : Array.isArray(r.recommendations)
        ? r.recommendations
        : [],
    topSkills:
      r.empathy_score != null
        ? [
            {
              skill: "Empathy",
              score: Math.round((r.empathy_score ?? 0) * 100),
            },
            {
              skill: "Professionalism",
              score: Math.round((r.professionalism_score ?? 0) * 100),
            },
            {
              skill: "Problem Solving",
              score: Math.round((r.problem_solving_score ?? 0) * 100),
            },
          ]
        : undefined,
  };
}

export default function AgentReports() {
  const [filter, setFilter] = useState<"all" | "weekly" | "monthly">("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedReport, setSelectedReport] = useState<Report | null>(null);
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchReports = async () => {
    try {
      setLoading(true);
      const response = await reportsApi.getMyReports();
      const rawData = response.data;
      const reportsArray = Array.isArray(rawData)
        ? rawData
        : rawData?.reports || [];
      setReports(reportsArray.map(mapReport));
    } catch (error) {
      console.error("Failed to fetch reports:", error);
      setReports([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchReports();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleViewDetails = async (report: Report) => {
    try {
      const response = await reportsApi.getReportDetail(report.id);
      setSelectedReport(mapReport(response.data));
    } catch {
      // Fall back to list-serializer data if detail fetch fails
      setSelectedReport(report);
    }
  };

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
            onViewDetails={handleViewDetails}
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
