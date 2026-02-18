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

export default function TeamReports() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedReport, setSelectedReport] = useState<Report | null>(null);
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchReports = async () => {
      try {
        setLoading(true);
        const response = await reportsApi.getAgentReports();
        const rawData = response.data;
        const data = Array.isArray(rawData) ? rawData : rawData?.reports || [];

        const mapped: Report[] = data.map((r: any) => ({
          id: r.id,
          agentName: r.agent_name || r.agent?.username || "Unknown",
          agentEmail: r.agent_email || r.agent?.email || "",
          type: r.type || "weekly",
          period: r.period || "",
          date: r.created_at ? new Date(r.created_at).toLocaleDateString() : "",
          overallScore: r.overall_score || 0,
          totalCalls: r.total_calls || 0,
          avgCallDuration: r.avg_call_duration || "0:00",
          trend: r.trend || "stable",
          strengths: r.strengths || [],
          improvements: r.improvements || [],
          topSkills: r.top_skills || [],
        }));

        setReports(mapped);
      } catch (error) {
        console.error("Failed to fetch reports:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchReports();
  }, []);

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

        <div className="space-y-3">
          {filteredReports.map((report) => (
            <ReportCard
              key={report.id}
              report={report}
              onView={() => setSelectedReport(report)}
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
