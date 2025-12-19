"use client";

import { useState } from "react";
import PerformanceHeader from "./agent/PerformanceHeader";
import PerformanceCharts from "./agent/PerformanceCharts";
import ReportFilters from "./agent/ReportFilters";
import ReportListItem, { Report } from "./agent/ReportListItem";
import ReportDetailModal from "./agent/ReportDetailModal";
import EmptyState from "./agent/EmptyState";

export default function AgentReports() {
  const [filter, setFilter] = useState<"all" | "weekly" | "monthly">("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedReport, setSelectedReport] = useState<Report | null>(null);

  const [reports] = useState<Report[]>([
    {
      id: 1,
      type: "weekly",
      period: "Week 50, 2024",
      date: "Dec 10 - Dec 16",
      score: 87,
      trend: "up",
      totalCalls: 24,
      avgDuration: 456,
      strengths: ["Active listening", "Empathy", "Problem resolution"],
      improvements: ["Call opening", "Product knowledge"],
      topSkills: [
        { skill: "Communication", score: 90 },
        { skill: "Problem Solving", score: 88 },
        { skill: "Empathy", score: 92 },
        { skill: "Product Knowledge", score: 79 },
      ],
    },
    {
      id: 2,
      type: "weekly",
      period: "Week 49, 2024",
      date: "Dec 3 - Dec 9",
      score: 84,
      trend: "up",
      totalCalls: 22,
      avgDuration: 432,
      strengths: ["Customer engagement", "Clear communication"],
      improvements: ["Follow-up questions", "Time management"],
      topSkills: [
        { skill: "Communication", score: 88 },
        { skill: "Problem Solving", score: 82 },
        { skill: "Empathy", score: 85 },
        { skill: "Product Knowledge", score: 80 },
      ],
    },
    {
      id: 3,
      type: "monthly",
      period: "November 2024",
      date: "Nov 1 - Nov 30",
      score: 82,
      trend: "stable",
      totalCalls: 89,
      avgDuration: 428,
      strengths: ["Professionalism", "Patience"],
      improvements: ["Technical knowledge", "Objection handling"],
      topSkills: [
        { skill: "Communication", score: 85 },
        { skill: "Problem Solving", score: 80 },
        { skill: "Empathy", score: 86 },
        { skill: "Product Knowledge", score: 75 },
      ],
    },
    {
      id: 4,
      type: "weekly",
      period: "Week 48, 2024",
      date: "Nov 26 - Dec 2",
      score: 81,
      trend: "down",
      totalCalls: 19,
      avgDuration: 445,
      strengths: ["Tone management", "Closing techniques"],
      improvements: ["Active listening", "Clarifying questions"],
      topSkills: [
        { skill: "Communication", score: 82 },
        { skill: "Problem Solving", score: 78 },
        { skill: "Empathy", score: 84 },
        { skill: "Product Knowledge", score: 80 },
      ],
    },
  ]);

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
