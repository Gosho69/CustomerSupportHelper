"use client";

import { useState, useEffect } from "react";
import { PageHeader, SearchInput } from "@/components/ui";
import { TeamMemberList, MemberDetailModal } from "@/components/team";
import { usersApi, callsApi, reportsApi } from "@/lib/api";
import { Users, Phone } from "lucide-react";
import { StatsCard } from "@/components/ui";

interface CallItem {
  id: number;
  agent: number;
  agent_name: string;
  call_date: string;
  duration: number | null;
  behavioral_score: number | null;
  created_at: string;
}

interface TeamMember {
  id: number;
  name: string;
  email: string;
  phone: string;
  joinDate: string;
  totalCalls: number;
  avgScore: number;
  avgCallDuration: string;
  trend: "up" | "down" | "stable";
  status: "active" | "on_break" | "offline";
  performanceData: { month: string; score: number }[];
}

function formatDuration(seconds: number | null | undefined): string {
  if (!seconds || seconds <= 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function MyTeam() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedMember, setSelectedMember] = useState<TeamMember | null>(null);
  const [selectedMemberCalls, setSelectedMemberCalls] = useState<CallItem[]>(
    [],
  );
  const [teamMembers, setTeamMembers] = useState<TeamMember[]>([]);
  const [teamCallsMap, setTeamCallsMap] = useState<Record<number, CallItem[]>>(
    {},
  );
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchTeam = async () => {
      try {
        setLoading(true);

        // Fetch subordinates and all team calls in parallel
        const [subordinatesRes, callsRes] = await Promise.all([
          usersApi.getSubordinates(),
          callsApi.getMyCalls(),
        ]);

        const subordinates: any[] = subordinatesRes.data || [];
        const allCalls: CallItem[] = callsRes.data || [];

        // Group calls by agent id
        const callsByAgent: Record<number, CallItem[]> = {};
        for (const call of allCalls) {
          if (!callsByAgent[call.agent]) callsByAgent[call.agent] = [];
          callsByAgent[call.agent].push(call);
        }
        setTeamCallsMap(callsByAgent);

        // Fetch latest reports for each agent in parallel
        const reportResults = await Promise.all(
          subordinates.map((user: any) =>
            reportsApi.getAgentReports(user.id).catch(() => ({ data: [] })),
          ),
        );

        const mapped: TeamMember[] = subordinates.map(
          (user: any, index: number) => {
            const agentCalls = callsByAgent[user.id] || [];
            const totalCalls = agentCalls.length;

            // Compute avg duration
            const durCalls = agentCalls.filter(
              (c) => c.duration && c.duration > 0,
            );
            const avgDurationSec =
              durCalls.length > 0
                ? durCalls.reduce((sum, c) => sum + (c.duration || 0), 0) /
                  durCalls.length
                : 0;
            const avgCallDuration = formatDuration(Math.round(avgDurationSec));

            // Compute avg score directly from calls' behavioral_score (0-100 scale)
            const scoredCalls = agentCalls.filter(
              (c: any) => c.behavioral_score != null && c.behavioral_score > 0,
            );
            const avgScore =
              scoredCalls.length > 0
                ? Math.round(
                    scoredCalls.reduce(
                      (sum: number, c: any) => sum + c.behavioral_score,
                      0,
                    ) / scoredCalls.length,
                  )
                : 0;

            // Use reports for trend and performance history
            // Note: average_behavioral_score is stored as 0-1 fraction, multiply by 100
            const rawReportData = reportResults[index]?.data;
            const reports: any[] = Array.isArray(rawReportData)
              ? rawReportData
              : rawReportData?.reports || [];
            const latestReport = reports[0];
            const rawTrend = latestReport?.behavioral_trend || "stable";
            const trend: "up" | "down" | "stable" =
              rawTrend === "improving"
                ? "up"
                : rawTrend === "declining"
                  ? "down"
                  : "stable";

            // Build performance chart data from reports (score is 0-1, convert to 0-100)
            const performanceData = reports
              .filter((r: any) => r.average_behavioral_score != null)
              .slice(0, 6)
              .reverse()
              .map((r: any) => ({
                month: new Date(r.start_date).toLocaleDateString("en-US", {
                  month: "short",
                  year: "2-digit",
                }),
                score: Math.round((r.average_behavioral_score ?? 0) * 100),
              }));

            return {
              id: user.id,
              name:
                `${user.first_name || ""} ${user.last_name || ""}`.trim() ||
                user.username,
              email: user.email || "",
              phone: user.phone || "",
              joinDate: user.date_joined
                ? new Date(user.date_joined).toLocaleDateString("en-US", {
                    year: "numeric",
                    month: "short",
                    day: "numeric",
                  })
                : "",
              totalCalls,
              avgScore,
              avgCallDuration,
              trend,
              status: "active" as const,
              performanceData,
            };
          },
        );

        setTeamMembers(mapped);
      } catch (error) {
        console.error("Failed to fetch team:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchTeam();
  }, []);

  const handleMemberClick = (member: TeamMember) => {
    setSelectedMember(member);
    setSelectedMemberCalls(teamCallsMap[member.id] || []);
  };

  const filteredMembers = teamMembers.filter((member) => {
    const matchesSearch =
      member.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      member.email.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesSearch;
  });

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div style={{ color: "var(--text-secondary)" }}>Loading team...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="My Team"
        subtitle="Manage and monitor your team's performance"
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <StatsCard
          icon={Users}
          label="Total Members"
          value={teamMembers.length}
        />
        <StatsCard
          icon={Phone}
          label="Total Calls"
          value={teamMembers.reduce((sum, m) => sum + m.totalCalls, 0)}
        />
      </div>

      <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
        <SearchInput
          value={searchQuery}
          onChange={setSearchQuery}
          placeholder="Search team members..."
          className="flex-1 w-full md:max-w-md"
        />
      </div>

      <TeamMemberList
        members={filteredMembers}
        onMemberClick={handleMemberClick}
      />

      <MemberDetailModal
        member={selectedMember}
        agentCalls={selectedMemberCalls}
        onClose={() => {
          setSelectedMember(null);
          setSelectedMemberCalls([]);
        }}
      />
    </div>
  );
}
