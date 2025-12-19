import { Phone } from "lucide-react";
import { Badge } from "@/components/ui";

export interface Call {
  id: number;
  agent_name: string;
  call_date: string;
  duration: number;
  created_at: string;
  sentiment?: "positive" | "neutral" | "negative";
  score?: number;
}

interface CallsTableProps {
  calls: Call[];
  loading: boolean;
  onViewDetails: (callId: number) => void;
}

export default function CallsTable({
  calls,
  loading,
  onViewDetails,
}: CallsTableProps) {
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  if (loading) {
    return (
      <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-12">
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
      <h2 className="text-xl font-bold text-white mb-6 flex items-center">
        <Phone className="w-5 h-5 mr-2 text-purple-400" />
        Call Recordings ({calls.length})
      </h2>

      <div className="space-y-3">
        {calls.map((call) => (
          <div
            key={call.id}
            className="bg-slate-900/50 rounded-xl p-5 hover:bg-slate-900/70 transition-all cursor-pointer"
            onClick={() => onViewDetails(call.id)}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4 flex-1">
                <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
                  <span className="text-white font-bold text-lg">
                    {call.agent_name.charAt(0)}
                  </span>
                </div>
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-1">
                    <p className="text-white font-semibold text-lg">
                      {call.agent_name}
                    </p>
                    {call.sentiment && (
                      <Badge
                        variant={
                          call.sentiment === "positive"
                            ? "green"
                            : call.sentiment === "neutral"
                            ? "blue"
                            : "red"
                        }
                        size="sm"
                      >
                        {call.sentiment}
                      </Badge>
                    )}
                  </div>
                  <p className="text-sm text-gray-400">
                    {formatDate(call.call_date)}
                  </p>
                </div>
              </div>

              <div className="flex items-center space-x-6">
                <div className="text-center">
                  <p className="text-gray-400 text-xs mb-1">Duration</p>
                  <p className="text-white font-semibold">
                    {formatDuration(call.duration)}
                  </p>
                </div>
                {call.score && (
                  <div className="text-center">
                    <p className="text-gray-400 text-xs mb-1">Score</p>
                    <p className="text-green-400 font-bold text-lg">
                      {call.score}%
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
