import {
  Phone,
  Mail,
  Calendar,
  Eye,
  TrendingUp,
  TrendingDown,
} from "lucide-react";
import { StatusIndicator } from "@/components/ui";

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
}

interface TeamMemberCardProps {
  member: TeamMember;
  onClick: () => void;
}

export default function TeamMemberCard({
  member,
  onClick,
}: TeamMemberCardProps) {
  const getScoreColor = (score: number) => {
    if (score >= 85) return "text-green-400";
    if (score >= 70) return "text-yellow-400";
    return "text-red-400";
  };

  const getTrendIcon = (trend: "up" | "down" | "stable") => {
    switch (trend) {
      case "up":
        return <TrendingUp className="w-4 h-4 text-green-400" />;
      case "down":
        return <TrendingDown className="w-4 h-4 text-red-400" />;
      default:
        return <div className="w-4 h-4 bg-gray-400 rounded-full" />;
    }
  };

  return (
    <div
      className="bg-slate-900/50 rounded-xl p-5 hover:bg-slate-900/70 transition-all cursor-pointer"
      onClick={onClick}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4 flex-1">
          <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
            <span className="text-white font-bold text-lg">
              {member.name.charAt(0)}
            </span>
          </div>
          <div className="flex-1">
            <div className="flex items-center space-x-3 mb-1">
              <p className="text-white font-semibold text-lg">{member.name}</p>
              <StatusIndicator status={member.status} />
            </div>
            <div className="flex items-center space-x-4 text-sm text-gray-400">
              <span className="flex items-center">
                <Mail className="w-3 h-3 mr-1" />
                {member.email}
              </span>
              <span className="flex items-center">
                <Phone className="w-3 h-3 mr-1" />
                {member.phone}
              </span>
              <span className="flex items-center">
                <Calendar className="w-3 h-3 mr-1" />
                Joined {member.joinDate}
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-6">
          <div className="text-center">
            <p className="text-gray-400 text-xs mb-1">Calls</p>
            <p className="text-white font-semibold">{member.totalCalls}</p>
          </div>
          <div className="text-center">
            <p className="text-gray-400 text-xs mb-1">Avg Score</p>
            <p
              className={`font-bold text-lg ${getScoreColor(member.avgScore)}`}
            >
              {member.avgScore}%
            </p>
          </div>
          <div className="text-center">
            <p className="text-gray-400 text-xs mb-1">Avg Duration</p>
            <p className="text-white font-semibold">{member.avgCallDuration}</p>
          </div>
          <div className="flex flex-col items-center">
            <p className="text-gray-400 text-xs mb-1">Trend</p>
            {getTrendIcon(member.trend)}
          </div>
          <button className="p-2 bg-purple-500/20 hover:bg-purple-500/30 rounded-lg transition-colors">
            <Eye className="w-5 h-5 text-purple-400" />
          </button>
        </div>
      </div>
    </div>
  );
}
