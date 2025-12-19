import { Download } from "lucide-react";
import { SearchInput } from "@/components/ui";

interface CallsFiltersProps {
  searchQuery: string;
  setSearchQuery: (term: string) => void;
  filterDuration: string;
  setFilterDuration: (duration: string) => void;
}

export default function CallsFilters({
  searchQuery,
  setSearchQuery,
  filterDuration,
  setFilterDuration,
}: CallsFiltersProps) {
  return (
    <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
      <SearchInput
        value={searchQuery}
        onChange={setSearchQuery}
        placeholder="Search by date..."
        className="flex-1 w-full md:max-w-md"
      />

      <div className="flex gap-3">
        <select
          value={filterDuration}
          onChange={(e) => setFilterDuration(e.target.value)}
          className="px-4 py-3 bg-slate-800/50 border border-white/10 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
        >
          <option value="all">All Durations</option>
          <option value="short">Short (&lt; 5 min)</option>
          <option value="medium">Medium (5-10 min)</option>
          <option value="long">Long (&gt; 10 min)</option>
        </select>

        <button className="px-6 py-3 bg-slate-700/50 hover:bg-slate-700 text-white rounded-xl font-semibold transition-all flex items-center space-x-2">
          <Download className="w-5 h-5" />
          <span>Export</span>
        </button>
      </div>
    </div>
  );
}
