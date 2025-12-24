import { Download } from "lucide-react";
import { SearchInput } from "@/components/ui";

interface UsersFiltersProps {
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  filterRole: string;
  setFilterRole: (role: string) => void;
}

export default function UsersFilters({
  searchQuery,
  setSearchQuery,
  filterRole,
  setFilterRole,
}: UsersFiltersProps) {
  return (
    <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
      <SearchInput
        value={searchQuery}
        onChange={setSearchQuery}
        placeholder="Search users..."
        className="flex-1 w-full md:max-w-md"
      />

      <div className="flex gap-3">
        <select
          value={filterRole}
          onChange={(e) => setFilterRole(e.target.value)}
          className="px-4 py-3 bg-slate-800/50 border border-white/10 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
        >
          <option value="all">All Roles</option>
          <option value="admin">Admin</option>
          <option value="head_of_department">Head of Department</option>
          <option value="agent">Agent</option>
        </select>

        <button className="px-6 py-3 bg-slate-700/50 hover:bg-slate-700 text-white rounded-xl font-semibold transition-all flex items-center space-x-2">
          <Download className="w-5 h-5" />
          <span>Export</span>
        </button>
      </div>
    </div>
  );
}
