"use client";

import { useState, useEffect } from "react";
import { FileText } from "lucide-react";
import { PageHeader, SearchInput, StatsCard } from "@/components/ui";
import { ReportCard, ReportDetailModal } from "@/components/reports";
import { reportsApi } from "@/lib/api";

interface Report {
  id: number;
  agentName: string;
  agentEmail: string;
  type: "weekly" | "monthly";
  period: string;
  date: string;
  overallScore: number;
  totalCalls: number;
  avgCallDuration: string;
  trend: "up" | "down" | "stable";
  strengths: string[];
  improvements: string[];
  topSkills: { skill: string; score: number }[];
}

function formatDuration(seconds: number | null | undefined): string {
  if (!seconds || seconds <= 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function mapTrend(raw: string | null | undefined): "up" | "down" | "stable" {
  if (raw === "improving") return "up";
  if (raw === "declining") return "down";
  return "stable";
}

function mapReport(r: any): Report {
  const startDate = r.start_date
    ? new Date(r.start_date).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      })
    : "";
  const endDate = r.end_date
    ? new Date(r.end_date).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      })
    : "";

  const topSkills: { skill: string; score: number }[] = [];
  if (r.empathy_score != null)
    topSkills.push({ skill: "Empathy", score: Math.round(r.empathy_score) });
  if (r.professionalism_score != null)
    topSkills.push({
      skill: "Professionalism",
      score: Math.round(r.professionalism_score),
    });
  if (r.problem_solving_score != null)
    topSkills.push({
      skill: "Problem Solving",
      score: Math.round(r.problem_solving_score),
    });

  return {
    id: r.id,
    agentName: r.agent_name || r.agent_username || "Unknown",
    agentEmail: r.agent_email || r.agent_username || "",
    type: (r.report_type as "weekly" | "monthly") || "weekly",
    period: startDate && endDate ? `${startDate} – ${endDate}` : startDate,
    date: r.generated_at
      ? new Date(r.generated_at).toLocaleDateString("en-US", {
          year: "numeric",
          month: "short",
          day: "numeric",
        })
      : "",
    overallScore: Math.round(r.average_behavioral_score ?? 0),
    totalCalls: r.total_calls ?? 0,
    avgCallDuration: formatDuration(r.average_call_duration),
    trend: mapTrend(r.behavioral_trend),
    strengths: Array.isArray(r.strengths) ? r.strengths : [],
    improvements: Array.isArray(r.weaknesses)
      ? r.weaknesses
      : Array.isArray(r.recommendations)
        ? r.recommendations
        : [],
    topSkills,
  };
}

export default function TeamReports() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedReport, setSelectedReport] = useState<Report | null>(null);
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);

  useEffect(() => {
    const fetchReports = async () => {
      try {
        setLoading(true);
        const response = await reportsApi.getAgentReports();
        const rawData = response.data;
        const data = Array.isArray(rawData) ? rawData : rawData?.reports || [];
        setReports(data.map(mapReport));
      } catch (error) {
        console.error("Failed to fetch reports:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchReports();
  }, []);

  // Fetch full report detail (with all fields) when user clicks View
  const handleViewReport = async (reportId: number) => {
    try {
      setLoadingDetail(true);
      const response = await reportsApi.getReportDetail(reportId);
      setSelectedReport(mapReport(response.data));
    } catch (error) {
      console.error("Failed to fetch report detail:", error);
      // Fallback to list data
      const fallback = reports.find((r) => r.id === reportId) || null;
      setSelectedReport(fallback);
    } finally {
      setLoadingDetail(false);
    }
  };

  const filteredReports = reports.filter((report) => {
    const matchesSearch =
      report.agentName.toLowerCase().includes(searchQuery.toLowerCase()) ||
      report.agentEmail.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesSearch;
  });

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div style={{ color: "var(--text-secondary)" }}>Loading reports...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Team Reports"
        subtitle="View and analyze your team's performance reports"
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <StatsCard
          icon={FileText}
          iconColor=""
          label="Total Reports"
          value={reports.length}
        />
        <StatsCard
          icon={FileText}
          iconColor=""
          label="Avg Team Score"
          value={
            reports.length > 0
              ? `${Math.round(reports.reduce((sum, r) => sum + r.overallScore, 0) / reports.length)}%`
              : "N/A"
          }
        />
      </div>

      <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
        <SearchInput
          value={searchQuery}
          onChange={setSearchQuery}
          placeholder="Search reports..."
          className="flex-1 w-full md:max-w-md"
        />
      </div>

      <div
        className="rounded-lg p-6"
        style={{ background: "#ffffff", border: "1px solid var(--border)" }}
      >
        <h2
          className="text-xl font-bold mb-6 flex items-center"
          style={{ color: "var(--text-primary)" }}
        >
          <FileText
            className="w-5 h-5 mr-2"
            style={{ color: "var(--accent)" }}
          />
          Reports ({filteredReports.length})
        </h2>

        {loadingDetail && (
          <div
            className="text-center py-4"
            style={{ color: "var(--text-secondary)" }}
          >
            Loading report details...
          </div>
        )}

        <div className="space-y-3">
          {filteredReports.map((report) => (
            <ReportCard
              key={report.id}
              report={report}
              onView={() => handleViewReport(report.id)}
            />
          ))}
        </div>
      </div>

      <ReportDetailModal
        report={selectedReport}
        onClose={() => setSelectedReport(null)}
      />
    </div>
  );
}
