import { Download } from "lucide-react";
import { SearchInput } from "@/components/ui";

interface CompaniesFiltersProps {
  searchQuery: string;
  setSearchQuery: (query: string) => void;
}

export default function CompaniesFilters({
  searchQuery,
  setSearchQuery,
}: CompaniesFiltersProps) {
  return (
    <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
      <SearchInput
        value={searchQuery}
        onChange={setSearchQuery}
        placeholder="Search companies..."
        className="flex-1 w-full md:max-w-md"
      />

      <div className="flex gap-3">
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
