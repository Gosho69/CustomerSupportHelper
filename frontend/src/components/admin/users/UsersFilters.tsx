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
        <div className="relative">
          <select
            value={filterRole}
            onChange={(e) => setFilterRole(e.target.value)}
            className="px-4 py-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 appearance-none cursor-pointer pr-10"
            style={{
              background: "#ffffff",
              border: "1px solid var(--border)",
              color: "var(--text-primary)",
            }}
          >
            <option value="all">All Roles</option>
            <option value="admin">Admin</option>
            <option value="head_of_department">Head of Department</option>
            <option value="agent">Agent</option>
          </select>
          <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
            <svg
              className="w-5 h-5"
              style={{ color: "var(--text-secondary)" }}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 9l-7 7-7-7"
              />
            </svg>
          </div>
        </div>

        <button
          className="px-6 py-3 hover:bg-gray-50 rounded-lg font-semibold transition-all flex items-center space-x-2"
          style={{
            background: "#ffffff",
            border: "1px solid var(--border)",
            color: "var(--text-primary)",
          }}
        >
          <Download className="w-5 h-5" />
          <span>Export</span>
        </button>
      </div>
    </div>
  );
}
