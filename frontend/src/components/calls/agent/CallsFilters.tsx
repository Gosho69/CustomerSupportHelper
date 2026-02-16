import { SearchInput, InlineSelect } from "@/components/ui";

interface CallsFiltersProps {
  searchQuery: string;
  setSearchQuery: (term: string) => void;
  filterDuration: string;
  setFilterDuration: (duration: string) => void;
}

const durationOptions = [
  { value: "all", label: "All Durations" },
  { value: "short", label: "Short (< 5 min)" },
  { value: "medium", label: "Medium (5-10 min)" },
  { value: "long", label: "Long (> 10 min)" },
];

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

      <InlineSelect
        options={durationOptions}
        value={filterDuration}
        onChange={setFilterDuration}
      />
    </div>
  );
}
