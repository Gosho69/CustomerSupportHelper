"use client";

import { useEffect, useState } from "react";
import { AllCalls } from "@/components/calls";
import {
  Phone,
  Calendar,
  Clock,
  Search,
  Filter,
  Download,
  TrendingUp,
  AlertCircle,
  Upload,
} from "lucide-react";
import { api } from "@/lib/api";
import CallDetailModal from "@/components/CallDetailModal";

interface Call {
  id: number;
  agent_name: string;
  call_date: string;
  duration: number;
  created_at: string;
}

export default function CallsPage() {
  const [userRole, setUserRole] = useState<
    "agent" | "head_of_department" | "admin"
  >("agent");

  useEffect(() => {
    // Get user role from localStorage for demo mode
    const storedRole = localStorage.getItem("demo_role") as
      | "agent"
      | "head_of_department"
      | "admin"
      | null;
    if (storedRole) {
      setUserRole(storedRole);
    }
  }, []);

  // If user is head of department, show the new AllCalls component
  if (userRole === "head_of_department") {
    return <AllCalls />;
  }

  // For agents, show the existing agent calls view
  return <AgentCallsView />;
}

function AgentCallsView() {
  const [calls, setCalls] = useState<Call[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedCallId, setSelectedCallId] = useState<number | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [filterDuration, setFilterDuration] = useState<string>("all");

  useEffect(() => {
    fetchCalls();
  }, []);

  const fetchCalls = async () => {
    try {
      setLoading(true);
      // Mock data for demo - replace with: const response = await api.calls.list();

      const mockCalls = [
        // Agent only sees their own calls
        {
          id: 1,
          agent_name: "You",
          call_date: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
          duration: 456,
          created_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
        },
        {
          id: 2,
          agent_name: "You",
          call_date: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
          duration: 723,
          created_at: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
        },
        {
          id: 3,
          agent_name: "You",
          call_date: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString(),
          duration: 312,
          created_at: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString(),
        },
        {
          id: 4,
          agent_name: "You",
          call_date: new Date(Date.now() - 72 * 60 * 60 * 1000).toISOString(),
          duration: 589,
          created_at: new Date(Date.now() - 72 * 60 * 60 * 1000).toISOString(),
        },
      ];
      setCalls(mockCalls);
    } catch (error) {
      console.error("Failed to fetch calls:", error);
    } finally {
      setLoading(false);
    }
  };

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

  const filteredCalls = calls.filter((call) => {
    const matchesSearch = call.agent_name
      .toLowerCase()
      .includes(searchTerm.toLowerCase());
    const matchesDuration =
      filterDuration === "all" ||
      (filterDuration === "short" && call.duration < 300) ||
      (filterDuration === "medium" &&
        call.duration >= 300 &&
        call.duration < 600) ||
      (filterDuration === "long" && call.duration >= 600);
    return matchesSearch && matchesDuration;
  });

  const stats = {
    total: calls.length,
    avgDuration:
      calls.length > 0
        ? Math.round(
            calls.reduce((sum, call) => sum + call.duration, 0) / calls.length
          )
        : 0,
    today: calls.filter(
      (call) =>
        new Date(call.call_date).toDateString() === new Date().toDateString()
    ).length,
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">My Calls</h1>
          <p className="text-gray-400 mt-1">
            View and analyze all your call recordings
          </p>
        </div>
        <button className="px-6 py-3 bg-gradient-to-r from-blue-600 to-cyan-600 text-white font-medium rounded-lg hover:from-blue-700 hover:to-cyan-700 transition-all duration-200 shadow-lg shadow-blue-500/30 flex items-center space-x-2">
          <Upload className="w-5 h-5" />
          <span>Upload Call</span>
        </button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Total Calls</p>
              <p className="text-3xl font-bold text-white mt-1">
                {stats.total}
              </p>
            </div>
            <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center">
              <Phone className="w-6 h-6 text-blue-400" />
            </div>
          </div>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Avg Duration</p>
              <p className="text-3xl font-bold text-white mt-1">
                {formatDuration(stats.avgDuration)}
              </p>
            </div>
            <div className="w-12 h-12 bg-cyan-500/20 rounded-lg flex items-center justify-center">
              <Clock className="w-6 h-6 text-cyan-400" />
            </div>
          </div>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Today</p>
              <p className="text-3xl font-bold text-white mt-1">
                {stats.today}
              </p>
            </div>
            <div className="w-12 h-12 bg-purple-500/20 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-6 h-6 text-purple-400" />
            </div>
          </div>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search calls..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-3 bg-slate-900/50 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20"
            />
          </div>
          <div className="flex items-center gap-2">
            <Filter className="w-5 h-5 text-gray-400" />
            <select
              value={filterDuration}
              onChange={(e) => setFilterDuration(e.target.value)}
              className="px-4 py-3 bg-slate-900/50 border border-white/10 rounded-lg text-white focus:outline-none focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20"
            >
              <option value="all">All Durations</option>
              <option value="short">Short (&lt; 5 min)</option>
              <option value="medium">Medium (5-10 min)</option>
              <option value="long">Long (&gt; 10 min)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Calls Table */}
      <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl overflow-hidden">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          </div>
        ) : filteredCalls.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-gray-400">
            <AlertCircle className="w-12 h-12 mb-4" />
            <p className="text-lg">No calls found</p>
            <p className="text-sm mt-1">Try adjusting your search criteria</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-slate-900/50 border-b border-white/10">
                <tr>
                  <th className="px-6 py-4 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                    Call ID
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                    Date & Time
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                    Duration
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/10">
                {filteredCalls.map((call) => (
                  <tr
                    key={call.id}
                    className="hover:bg-white/5 transition-colors cursor-pointer"
                    onClick={() => setSelectedCallId(call.id)}
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="text-sm font-medium text-white">
                        #{call.id}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center text-sm text-gray-300">
                        <Calendar className="w-4 h-4 mr-2 text-gray-400" />
                        {formatDate(call.call_date)}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center text-sm text-gray-300">
                        <Clock className="w-4 h-4 mr-2 text-gray-400" />
                        {formatDuration(call.duration)}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedCallId(call.id);
                        }}
                        className="px-4 py-2 bg-gradient-to-r from-blue-600 to-cyan-600 text-white text-sm font-medium rounded-lg hover:from-blue-700 hover:to-cyan-700 transition-all duration-200 shadow-lg shadow-blue-500/30"
                      >
                        View Details
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Call Detail Modal */}
      {selectedCallId && (
        <CallDetailModal
          callId={selectedCallId}
          onClose={() => setSelectedCallId(null)}
        />
      )}
    </div>
  );
}
