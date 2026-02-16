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
    </div>
  );
}
