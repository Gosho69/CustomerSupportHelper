"use client";

import { useState } from "react";
import {
  FileText,
  Award,
  TrendingUp,
  TrendingDown,
  BarChart3,
  Download,
} from "lucide-react";
import { PageHeader, SearchInput, StatsCard } from "@/components/ui";
import { TrendLineChart } from "@/components/charts";
import { ReportCard, ReportDetailModal } from "@/components/reports";

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
  const [filterType, setFilterType] = useState<"all" | "weekly" | "monthly">(
    "all"
  );
  const [selectedReport, setSelectedReport] = useState<Report | null>(null);

  const [reports] = useState<Report[]>([
    {
      id: 1,
      agentName: "John Smith",
      agentEmail: "john.smith@example.com",
      type: "weekly",
      period: "Dec 10 - Dec 16, 2024",
      date: "Dec 17, 2024",
      overallScore: 92,
      totalCalls: 45,
      avgCallDuration: "8:45",
      trend: "up",
      strengths: [
        "Excellent problem resolution",
        "Strong product knowledge",
        "Professional communication",
      ],
      improvements: [
        "Reduce call transfer rate",
        "Improve first call resolution",
      ],
      topSkills: [
        { skill: "Communication", score: 95 },
        { skill: "Problem Solving", score: 90 },
        { skill: "Product Knowledge", score: 92 },
        { skill: "Empathy", score: 88 },
      ],
    },
    {
      id: 2,
      agentName: "Sarah Johnson",
      agentEmail: "sarah.j@example.com",
      type: "weekly",
      period: "Dec 10 - Dec 16, 2024",
      date: "Dec 17, 2024",
      overallScore: 88,
      totalCalls: 38,
      avgCallDuration: "7:30",
      trend: "up",
      strengths: [
        "Great customer empathy",
        "Fast response time",
        "Clear communication",
      ],
      improvements: ["Work on technical knowledge", "Reduce call duration"],
      topSkills: [
        { skill: "Communication", score: 90 },
        { skill: "Empathy", score: 95 },
        { skill: "Efficiency", score: 85 },
        { skill: "Problem Solving", score: 82 },
      ],
    },
    {
      id: 3,
      agentName: "Mike Wilson",
      agentEmail: "mike.w@example.com",
      type: "monthly",
      period: "November 2024",
      date: "Dec 1, 2024",
      overallScore: 76,
      totalCalls: 168,
      avgCallDuration: "9:15",
      trend: "down",
      strengths: ["Detailed documentation", "Patient with customers"],
      improvements: [
        "Improve closing techniques",
        "Reduce average handle time",
        "Work on upselling skills",
      ],
      topSkills: [
        { skill: "Communication", score: 78 },
        { skill: "Problem Solving", score: 75 },
        { skill: "Product Knowledge", score: 80 },
        { skill: "Empathy", score: 85 },
      ],
    },
    {
      id: 4,
      agentName: "Emily Davis",
      agentEmail: "emily.d@example.com",
      type: "weekly",
      period: "Dec 10 - Dec 16, 2024",
      date: "Dec 17, 2024",
      overallScore: 85,
      totalCalls: 31,
      avgCallDuration: "8:00",
      trend: "stable",
      strengths: [
        "Consistent performance",
        "Good product knowledge",
        "Professional demeanor",
      ],
      improvements: ["Increase call volume", "Work on closing rate"],
      topSkills: [
        { skill: "Communication", score: 85 },
        { skill: "Problem Solving", score: 84 },
        { skill: "Product Knowledge", score: 88 },
        { skill: "Empathy", score: 86 },
      ],
    },
    {
      id: 5,
      agentName: "Robert Brown",
      agentEmail: "robert.b@example.com",
      type: "monthly",
      period: "November 2024",
      date: "Dec 1, 2024",
      overallScore: 81,
      totalCalls: 152,
      avgCallDuration: "7:45",
      trend: "up",
      strengths: [
        "Improving steadily",
        "Good technical skills",
        "Positive attitude",
      ],
      improvements: ["Work on empathy", "Improve follow-up processes"],
      topSkills: [
        { skill: "Communication", score: 80 },
        { skill: "Problem Solving", score: 82 },
        { skill: "Product Knowledge", score: 85 },
        { skill: "Efficiency", score: 78 },
      ],
    },
  ]);

  const filteredReports = reports.filter((report) => {
    const matchesSearch =
      report.agentName.toLowerCase().includes(searchQuery.toLowerCase()) ||
      report.agentEmail.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesFilter = filterType === "all" || report.type === filterType;
    return matchesSearch && matchesFilter;
  });

  const teamTrendData = [
    { month: "Jul", avgScore: 82 },
    { month: "Aug", avgScore: 83 },
    { month: "Sep", avgScore: 84 },
    { month: "Oct", avgScore: 85 },
    { month: "Nov", avgScore: 84 },
    { month: "Dec", avgScore: 86 },
  ];

  const avgTeamScore = Math.round(
    reports.reduce((sum, r) => sum + r.overallScore, 0) / reports.length
  );
  const improvingCount = reports.filter((r) => r.trend === "up").length;
  const needAttentionCount = reports.filter((r) => r.trend === "down").length;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Team Reports"
        subtitle="View and analyze your team's performance reports"
      />

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <StatsCard
          icon={FileText}
          iconColor="bg-blue-500/20 text-blue-400"
          label="Total Reports"
          value={reports.length}
        />
        <StatsCard
          icon={Award}
          iconColor="bg-green-500/20 text-green-400"
          label="Avg Team Score"
          value={`${avgTeamScore}%`}
        />
        <StatsCard
          icon={TrendingUp}
          iconColor="bg-cyan-500/20 text-cyan-400"
          label="Improving"
          value={improvingCount}
        />
        <StatsCard
          icon={TrendingDown}
          iconColor="bg-red-500/20 text-red-400"
          label="Need Attention"
          value={needAttentionCount}
        />
      </div>

      <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
        <h2 className="text-xl font-bold text-white mb-6 flex items-center">
          <BarChart3 className="w-5 h-5 mr-2 text-purple-400" />
          Team Performance Trend
        </h2>
        <TrendLineChart data={teamTrendData} dataKey="avgScore" />
      </div>

      <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
        <SearchInput
          value={searchQuery}
          onChange={setSearchQuery}
          placeholder="Search reports..."
          className="flex-1 w-full md:max-w-md"
        />

        <div className="flex gap-3">
          <select
            value={filterType}
            onChange={(e) =>
              setFilterType(e.target.value as "all" | "weekly" | "monthly")
            }
            className="px-4 py-3 bg-slate-800/50 border border-white/10 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
          >
            <option value="all">All Reports</option>
            <option value="weekly">Weekly</option>
            <option value="monthly">Monthly</option>
          </select>

          <button className="px-6 py-3 bg-slate-700/50 hover:bg-slate-700 text-white rounded-xl font-semibold transition-all flex items-center space-x-2">
            <Download className="w-5 h-5" />
            <span>Export All</span>
          </button>
        </div>
      </div>

      <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
        <h2 className="text-xl font-bold text-white mb-6 flex items-center">
          <FileText className="w-5 h-5 mr-2 text-purple-400" />
          Reports ({filteredReports.length})
        </h2>

        <div className="space-y-3">
          {filteredReports.map((report) => (
            <ReportCard
              key={report.id}
              report={report}
              onView={() => setSelectedReport(report)}
              onDownload={() => console.log("Download report:", report.id)}
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
