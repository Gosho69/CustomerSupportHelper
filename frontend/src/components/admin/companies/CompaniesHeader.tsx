import { Plus } from "lucide-react";

interface CompaniesHeaderProps {
  onAddCompany: () => void;
}

export default function CompaniesHeader({
  onAddCompany,
}: CompaniesHeaderProps) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <h1
          className="text-3xl font-bold"
          style={{ color: "var(--text-primary)" }}
        >
          Companies
        </h1>
        <p className="mt-1" style={{ color: "var(--text-secondary)" }}>
          Manage all registered companies in the system
        </p>
      </div>
      <button
        onClick={onAddCompany}
        className="px-6 py-3 font-semibold rounded-lg transition-all flex items-center space-x-2"
        style={{ background: "var(--accent-bg)", color: "var(--accent)" }}
      >
        <Plus className="w-5 h-5" />
        <span>Add Company</span>
      </button>
    </div>
  );
}
