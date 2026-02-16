import { Users, Download } from "lucide-react";
import TeamMemberCard from "./TeamMemberCard";

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

interface TeamMemberListProps {
  members: TeamMember[];
  onMemberClick: (member: TeamMember) => void;
}

export default function TeamMemberList({
  members,
  onMemberClick,
}: TeamMemberListProps) {
  return (
    <div
      className="rounded-lg p-6"
      style={{
        background: "#ffffff",
        border: "1px solid var(--border)",
        borderRadius: "8px",
      }}
    >
      <div className="flex items-center justify-between mb-6">
        <h2
          className="text-xl font-bold flex items-center"
          style={{ color: "var(--text-primary)" }}
        >
          <Users className="w-5 h-5 mr-2" style={{ color: "var(--accent)" }} />
          Team Members ({members.length})
        </h2>
        <button
          className="px-4 py-2 rounded-lg transition-colors flex items-center space-x-2 hover:shadow-sm"
          style={{
            background: "var(--accent-bg)",
            color: "var(--text-primary)",
            border: "1px solid var(--border)",
          }}
        >
          <Download className="w-4 h-4" />
          <span>Export</span>
        </button>
      </div>

      <div className="space-y-3">
        {members.map((member) => (
          <TeamMemberCard
            key={member.id}
            member={member}
            onClick={() => onMemberClick(member)}
          />
        ))}
      </div>
    </div>
  );
}
