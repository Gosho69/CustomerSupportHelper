import { Filter, Search } from "lucide-react";

interface ReportFiltersProps {
  filter: "all" | "weekly" | "monthly";
  setFilter: (filter: "all" | "weekly" | "monthly") => void;
  searchQuery: string;
  setSearchQuery: (query: string) => void;
}

export default function ReportFilters({
  filter,
  setFilter,
  searchQuery,
  setSearchQuery,
}: ReportFiltersProps) {
  return (
    <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div className="flex items-center space-x-2">
          <Filter className="w-5 h-5 text-gray-400" />
          <div className="flex space-x-2">
            {["all", "weekly", "monthly"].map((type) => (
              <button
                key={type}
                onClick={() => setFilter(type as any)}
                className={`px-4 py-2 rounded-lg font-medium transition-all ${
                  filter === type
                    ? "bg-blue-600 text-white"
                    : "bg-slate-700/50 text-gray-400 hover:bg-slate-700"
                }`}
              >
                {type.charAt(0).toUpperCase() + type.slice(1)}
              </button>
            ))}
          </div>
        </div>
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search reports..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 pr-4 py-2 bg-slate-700/50 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>
    </div>
  );
}
