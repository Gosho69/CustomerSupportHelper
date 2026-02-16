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
    <div
      className="rounded-lg p-6"
      style={{ background: "#ffffff", border: "1px solid var(--border)" }}
    >
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div className="flex items-center space-x-2">
          <Filter
            className="w-5 h-5"
            style={{ color: "var(--text-secondary)" }}
          />
          <div className="flex space-x-2">
            {["all", "weekly", "monthly"].map((type) => (
              <button
                key={type}
                onClick={() => setFilter(type as any)}
                className={`px-4 py-2 rounded-lg font-medium transition-all ${
                  filter === type ? "text-white" : ""
                }`}
                style={
                  filter === type
                    ? { background: "var(--accent)" }
                    : {
                        background: "var(--accent-bg)",
                        color: "var(--text-secondary)",
                      }
                }
              >
                {type.charAt(0).toUpperCase() + type.slice(1)}
              </button>
            ))}
          </div>
        </div>
        <div className="relative">
          <Search
            className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5"
            style={{ color: "var(--text-secondary)" }}
          />
          <input
            type="text"
            placeholder="Search reports..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 pr-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#635bff]"
            style={{
              background: "var(--background)",
              border: "1px solid var(--border)",
              color: "var(--text-primary)",
            }}
          />
        </div>
      </div>
    </div>
  );
}
