"use client";

import { useState } from "react";
import { UserPlus } from "lucide-react";
import { PageHeader, SearchInput, Button } from "@/components/ui";
import {
  TeamStatsGrid,
  TeamSkillsChart,
  TeamMemberList,
  MemberDetailModal,
  AddMemberModal,
} from "@/components/team";

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
  const [filterStatus, setFilterStatus] = useState<
    "all" | "active" | "on_break" | "offline"
  >("all");
  const [selectedMember, setSelectedMember] = useState<TeamMember | null>(null);
  const [showAddMember, setShowAddMember] = useState(false);

  const [teamMembers] = useState<TeamMember[]>([
    {
      id: 1,
      name: "John Smith",
      email: "john.smith@example.com",
      phone: "+1 234 567 8901",
      joinDate: "Jan 15, 2024",
      totalCalls: 245,
      avgScore: 92,
      avgCallDuration: "8:45",
      trend: "up",
      status: "active",
      performanceData: [
        { month: "Jul", score: 85 },
        { month: "Aug", score: 88 },
        { month: "Sep", score: 87 },
        { month: "Oct", score: 90 },
        { month: "Nov", score: 91 },
        { month: "Dec", score: 92 },
      ],
    },
    {
      id: 2,
      name: "Sarah Johnson",
      email: "sarah.j@example.com",
      phone: "+1 234 567 8902",
      joinDate: "Feb 20, 2024",
      totalCalls: 198,
      avgScore: 88,
      avgCallDuration: "7:30",
      trend: "up",
      status: "active",
      performanceData: [
        { month: "Jul", score: 82 },
        { month: "Aug", score: 84 },
        { month: "Sep", score: 86 },
        { month: "Oct", score: 85 },
        { month: "Nov", score: 87 },
        { month: "Dec", score: 88 },
      ],
    },
    {
      id: 3,
      name: "Mike Wilson",
      email: "mike.w@example.com",
      phone: "+1 234 567 8903",
      joinDate: "Mar 10, 2024",
      totalCalls: 210,
      avgScore: 76,
      avgCallDuration: "9:15",
      trend: "down",
      status: "on_break",
      performanceData: [
        { month: "Jul", score: 85 },
        { month: "Aug", score: 83 },
        { month: "Sep", score: 80 },
        { month: "Oct", score: 78 },
        { month: "Nov", score: 77 },
        { month: "Dec", score: 76 },
      ],
    },
    {
      id: 4,
      name: "Emily Davis",
      email: "emily.d@example.com",
      phone: "+1 234 567 8904",
      joinDate: "Apr 5, 2024",
      totalCalls: 167,
      avgScore: 85,
      avgCallDuration: "8:00",
      trend: "stable",
      status: "active",
      performanceData: [
        { month: "Jul", score: 84 },
        { month: "Aug", score: 85 },
        { month: "Sep", score: 84 },
        { month: "Oct", score: 85 },
        { month: "Nov", score: 86 },
        { month: "Dec", score: 85 },
      ],
    },
    {
      id: 5,
      name: "Robert Brown",
      email: "robert.b@example.com",
      phone: "+1 234 567 8905",
      joinDate: "May 12, 2024",
      totalCalls: 189,
      avgScore: 81,
      avgCallDuration: "7:45",
      trend: "up",
      status: "active",
      performanceData: [
        { month: "Jul", score: 75 },
        { month: "Aug", score: 77 },
        { month: "Sep", score: 78 },
        { month: "Oct", score: 79 },
        { month: "Nov", score: 80 },
        { month: "Dec", score: 81 },
      ],
    },
    {
      id: 6,
      name: "Lisa Anderson",
      email: "lisa.a@example.com",
      phone: "+1 234 567 8906",
      joinDate: "Jun 18, 2024",
      totalCalls: 156,
      avgScore: 89,
      avgCallDuration: "8:20",
      trend: "up",
      status: "offline",
      performanceData: [
        { month: "Jul", score: 82 },
        { month: "Aug", score: 84 },
        { month: "Sep", score: 86 },
        { month: "Oct", score: 87 },
        { month: "Nov", score: 88 },
        { month: "Dec", score: 89 },
      ],
    },
  ]);

  const filteredMembers = teamMembers.filter((member) => {
    const matchesSearch =
      member.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      member.email.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesFilter =
      filterStatus === "all" || member.status === filterStatus;
    return matchesSearch && matchesFilter;
  });

  const teamSkillsData = [
    { skill: "Communication", value: 85 },
    { skill: "Problem Solving", value: 82 },
    { skill: "Product Knowledge", value: 88 },
    { skill: "Empathy", value: 90 },
    { skill: "Efficiency", value: 83 },
    { skill: "Closing", value: 79 },
  ];

  const avgPerformance = Math.round(
    teamMembers.reduce((sum, m) => sum + m.avgScore, 0) / teamMembers.length,
  );
  const totalCalls = teamMembers.reduce((sum, m) => sum + m.totalCalls, 0);
  const activeMembers = teamMembers.filter((m) => m.status === "active").length;

  return (
    <div className="space-y-6">
      <PageHeader
        title="My Team"
        subtitle="Manage and monitor your team's performance"
      />

      <TeamStatsGrid
        totalMembers={teamMembers.length}
        avgPerformance={avgPerformance}
        totalCalls={totalCalls}
        activeMembers={activeMembers}
      />

      <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
        <SearchInput
          value={searchQuery}
          onChange={setSearchQuery}
          placeholder="Search team members..."
          className="flex-1 w-full md:max-w-md"
        />

        <div className="flex gap-3">
          <div className="relative">
            <select
              value={filterStatus}
              onChange={(e) =>
                setFilterStatus(
                  e.target.value as "all" | "active" | "on_break" | "offline",
                )
              }
              className="px-4 py-3 rounded-lg appearance-none cursor-pointer pr-10 focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
              style={{
                background: "#ffffff",
                border: "1px solid var(--border)",
                color: "var(--text-primary)",
              }}
            >
              <option value="all">All Status</option>
              <option value="active">Active</option>
              <option value="on_break">On Break</option>
              <option value="offline">Offline</option>
            </select>
            <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
              <svg
                className="w-5 h-5"
                style={{ color: "var(--text-secondary)" }}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 9l-7 7-7-7"
                />
              </svg>
            </div>
          </div>

          <Button
            onClick={() => setShowAddMember(true)}
            icon={UserPlus}
            iconPosition="left"
          >
            Add Member
          </Button>
        </div>
      </div>

      <TeamSkillsChart data={teamSkillsData} />

      <TeamMemberList
        members={filteredMembers}
        onMemberClick={setSelectedMember}
      />

      <MemberDetailModal
        member={selectedMember}
        onClose={() => setSelectedMember(null)}
      />

      <AddMemberModal
        isOpen={showAddMember}
        onClose={() => setShowAddMember(false)}
        onAdd={(data) => {
          // Handle add member logic
          console.log("Adding member:", data);
        }}
      />
    </div>
  );
}
