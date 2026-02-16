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
    return "";
  };

  const getTrendIcon = (trend: "up" | "down" | "stable") => {
    switch (trend) {
      case "up":
        return (
          <TrendingUp
            className="w-4 h-4"
            style={{ color: "var(--success, #0caf60)" }}
          />
        );
      case "down":
        return (
          <TrendingDown
            className="w-4 h-4"
            style={{ color: "var(--error, #e53935)" }}
          />
        );
      default:
        return <div className="w-4 h-4 bg-gray-400 rounded-full" />;
    }
  };

  return (
    <div
      className="rounded-lg p-5 transition-all cursor-pointer hover:shadow-sm"
      onClick={onClick}
      style={{
        background: "#ffffff",
        border: "1px solid var(--border)",
      }}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4 flex-1">
          <div
            className="w-12 h-12 rounded-full flex items-center justify-center"
            style={{ background: "var(--accent-bg)" }}
          >
            <span
              className="font-bold text-lg"
              style={{ color: "var(--accent)" }}
            >
              {member.name.charAt(0)}
            </span>
          </div>
          <div className="flex-1">
            <div className="flex items-center space-x-3 mb-1">
              <p
                className="font-semibold text-lg"
                style={{ color: "var(--text-primary)" }}
              >
                {member.name}
              </p>
              <StatusIndicator status={member.status} />
            </div>
            <div
              className="flex items-center space-x-4 text-sm"
              style={{ color: "var(--text-secondary)" }}
            >
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
            <p
              className="text-xs mb-1"
              style={{ color: "var(--text-secondary)" }}
            >
              Calls
            </p>
            <p
              className="font-semibold"
              style={{ color: "var(--text-primary)" }}
            >
              {member.totalCalls}
            </p>
          </div>
          <div className="text-center">
            <p
              className="text-xs mb-1"
              style={{ color: "var(--text-secondary)" }}
            >
              Avg Score
            </p>
            <p
              className={`font-bold text-lg ${getScoreColor(member.avgScore)}`}
              style={{ color: "var(--text-primary)" }}
            >
              {member.avgScore}%
            </p>
          </div>
          <div className="text-center">
            <p
              className="text-xs mb-1"
              style={{ color: "var(--text-secondary)" }}
            >
              Avg Duration
            </p>
            <p
              className="font-semibold"
              style={{ color: "var(--text-primary)" }}
            >
              {member.avgCallDuration}
            </p>
          </div>
          <div className="flex flex-col items-center">
            <p
              className="text-xs mb-1"
              style={{ color: "var(--text-secondary)" }}
            >
              Trend
            </p>
            {getTrendIcon(member.trend)}
          </div>
          <button
            className="p-2 rounded-lg transition-colors"
            style={{ background: "var(--accent-bg)" }}
          >
            <Eye className="w-5 h-5" style={{ color: "var(--accent)" }} />
          </button>
        </div>
      </div>
    </div>
  );
}
