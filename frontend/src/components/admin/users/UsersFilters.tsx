import { SearchInput, InlineSelect } from "@/components/ui";

interface UsersFiltersProps {
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  filterRole: string;
  setFilterRole: (role: string) => void;
}

const roleOptions = [
  { value: "all", label: "All Roles" },
  { value: "admin", label: "Admin" },
  { value: "head_of_department", label: "Head of Department" },
  { value: "agent", label: "Agent" },
];

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

      <InlineSelect
        options={roleOptions}
        value={filterRole}
        onChange={setFilterRole}
      />
    </div>
  );
}
