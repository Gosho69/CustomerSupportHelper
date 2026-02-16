import { Users } from "lucide-react";
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
      <h2
        className="text-xl font-bold flex items-center mb-6"
        style={{ color: "var(--text-primary)" }}
      >
        <Users className="w-5 h-5 mr-2" style={{ color: "var(--accent)" }} />
        Team Members ({members.length})
      </h2>

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
