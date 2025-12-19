import { Phone, Mail, Calendar, TrendingUp, TrendingDown } from "lucide-react";
import { Modal, StatusIndicator } from "@/components/ui";
import { PerformanceAreaChart } from "@/components/charts";

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

interface MemberDetailModalProps {
  member: TeamMember | null;
  onClose: () => void;
}

export default function MemberDetailModal({
  member,
  onClose,
}: MemberDetailModalProps) {
  if (!member) return null;

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
    <Modal isOpen={!!member} onClose={onClose} size="4xl">
      <div className="flex items-start space-x-6 mb-8">
        <div className="w-20 h-20 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
          <span className="text-white font-bold text-3xl">
            {member.name.charAt(0)}
          </span>
        </div>
        <div className="flex-1">
          <div className="flex items-center space-x-3 mb-2">
            <h2 className="text-2xl font-bold text-white">{member.name}</h2>
            <StatusIndicator status={member.status} />
          </div>
          <div className="space-y-1 text-gray-400">
            <p className="flex items-center">
              <Mail className="w-4 h-4 mr-2" />
              {member.email}
            </p>
            <p className="flex items-center">
              <Phone className="w-4 h-4 mr-2" />
              {member.phone}
            </p>
            <p className="flex items-center">
              <Calendar className="w-4 h-4 mr-2" />
              Joined {member.joinDate}
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-4 mb-8">
        <div className="bg-slate-900/50 rounded-xl p-4">
          <p className="text-gray-400 text-sm mb-1">Total Calls</p>
          <p className="text-2xl font-bold text-white">{member.totalCalls}</p>
        </div>
        <div className="bg-slate-900/50 rounded-xl p-4">
          <p className="text-gray-400 text-sm mb-1">Avg Score</p>
          <p className={`text-2xl font-bold ${getScoreColor(member.avgScore)}`}>
            {member.avgScore}%
          </p>
        </div>
        <div className="bg-slate-900/50 rounded-xl p-4">
          <p className="text-gray-400 text-sm mb-1">Avg Duration</p>
          <p className="text-2xl font-bold text-white">
            {member.avgCallDuration}
          </p>
        </div>
        <div className="bg-slate-900/50 rounded-xl p-4">
          <p className="text-gray-400 text-sm mb-1">Trend</p>
          <div className="flex items-center space-x-2 mt-2">
            {getTrendIcon(member.trend)}
            <span className="text-white font-semibold capitalize">
              {member.trend}
            </span>
          </div>
        </div>
      </div>

      <div className="bg-slate-900/50 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4">
          Performance History
        </h3>
        <PerformanceAreaChart data={member.performanceData} />
      </div>
    </Modal>
  );
}
