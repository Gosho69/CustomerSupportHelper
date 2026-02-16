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
    <Modal isOpen={!!member} onClose={onClose} size="4xl">
      <div className="flex items-start space-x-6 mb-8">
        <div
          className="w-20 h-20 rounded-full flex items-center justify-center"
          style={{ background: "var(--accent-bg)" }}
        >
          <span
            className="font-bold text-3xl"
            style={{ color: "var(--accent)" }}
          >
            {member.name.charAt(0)}
          </span>
        </div>
        <div className="flex-1">
          <div className="flex items-center space-x-3 mb-2">
            <h2
              className="text-2xl font-bold"
              style={{ color: "var(--text-primary)" }}
            >
              {member.name}
            </h2>
            <StatusIndicator status={member.status} />
          </div>
          <div className="space-y-1" style={{ color: "var(--text-secondary)" }}>
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
        <div
          className="rounded-lg p-4"
          style={{
            background: "#ffffff",
            border: "1px solid var(--border)",
          }}
        >
          <p
            className="text-sm mb-1"
            style={{ color: "var(--text-secondary)" }}
          >
            Total Calls
          </p>
          <p
            className="text-2xl font-bold"
            style={{ color: "var(--text-primary)" }}
          >
            {member.totalCalls}
          </p>
        </div>
        <div
          className="rounded-lg p-4"
          style={{
            background: "#ffffff",
            border: "1px solid var(--border)",
          }}
        >
          <p
            className="text-sm mb-1"
            style={{ color: "var(--text-secondary)" }}
          >
            Avg Score
          </p>
          <p
            className={`text-2xl font-bold ${getScoreColor(member.avgScore)}`}
            style={{ color: "var(--text-primary)" }}
          >
            {member.avgScore}%
          </p>
        </div>
        <div
          className="rounded-lg p-4"
          style={{
            background: "#ffffff",
            border: "1px solid var(--border)",
          }}
        >
          <p
            className="text-sm mb-1"
            style={{ color: "var(--text-secondary)" }}
          >
            Avg Duration
          </p>
          <p
            className="text-2xl font-bold"
            style={{ color: "var(--text-primary)" }}
          >
            {member.avgCallDuration}
          </p>
        </div>
        <div
          className="rounded-lg p-4"
          style={{
            background: "#ffffff",
            border: "1px solid var(--border)",
          }}
        >
          <p
            className="text-sm mb-1"
            style={{ color: "var(--text-secondary)" }}
          >
            Trend
          </p>
          <div className="flex items-center space-x-2 mt-2">
            {getTrendIcon(member.trend)}
            <span
              className="font-semibold capitalize"
              style={{ color: "var(--text-primary)" }}
            >
              {member.trend}
            </span>
          </div>
        </div>
      </div>

      <div
        className="rounded-lg p-6"
        style={{
          background: "var(--background)",
          border: "1px solid var(--border)",
        }}
      >
        <h3
          className="text-lg font-semibold mb-4"
          style={{ color: "var(--text-primary)" }}
        >
          Performance History
        </h3>
        <PerformanceAreaChart data={member.performanceData} />
      </div>
    </Modal>
  );
}
