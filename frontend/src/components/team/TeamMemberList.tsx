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
    <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-white flex items-center">
          <Users className="w-5 h-5 mr-2 text-purple-400" />
          Team Members ({members.length})
        </h2>
        <button className="px-4 py-2 bg-slate-700/50 hover:bg-slate-700 text-white rounded-lg transition-colors flex items-center space-x-2">
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
