"use client";

import { useState, useEffect } from "react";
import { PageHeader, SearchInput } from "@/components/ui";
import { TeamMemberList, MemberDetailModal } from "@/components/team";
import { usersApi } from "@/lib/api";
import { Users, Phone } from "lucide-react";
import { StatsCard } from "@/components/ui";

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

export default function MyTeam() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedMember, setSelectedMember] = useState<TeamMember | null>(null);
  const [teamMembers, setTeamMembers] = useState<TeamMember[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchTeam = async () => {
      try {
        setLoading(true);
        const response = await usersApi.getSubordinates();
        const subordinates = response.data || [];

        const mapped: TeamMember[] = subordinates.map((user: any) => ({
          id: user.id,
          name:
            `${user.first_name || ""} ${user.last_name || ""}`.trim() ||
            user.username,
          email: user.email || "",
          phone: user.phone || "",
          joinDate: "",
          totalCalls: 0,
          avgScore: 0,
          avgCallDuration: "0:00",
          trend: "stable" as const,
          status: "active" as const,
          performanceData: [],
        }));

        setTeamMembers(mapped);
      } catch (error) {
        console.error("Failed to fetch team:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchTeam();
  }, []);

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
        onMemberClick={setSelectedMember}
      />

      <MemberDetailModal
        member={selectedMember}
        onClose={() => setSelectedMember(null)}
      />
    </div>
  );
}
