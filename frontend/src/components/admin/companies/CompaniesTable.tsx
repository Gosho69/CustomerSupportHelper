import { Building2, Edit, Trash2, Calendar, Users } from "lucide-react";
import { Badge } from "@/components/ui";

interface Company {
  id: number;
  name: string;
  industry: string;
  address: string;
  phone: string;
  purpose: string;
  employees: number;
  created_at: string;
  status: "active" | "inactive";
}

interface CompaniesTableProps {
  companies: Company[];
  onEdit: (company: Company) => void;
  onDelete: (id: number) => void;
}

export default function CompaniesTable({
  companies,
  onEdit,
  onDelete,
}: CompaniesTableProps) {
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  };

  return (
    <div
      className="rounded-lg p-6"
      style={{
        background: "#ffffff",
        border: "1px solid var(--border)",
      }}
    >
      <h2
        className="text-xl font-bold mb-6 flex items-center"
        style={{ color: "var(--text-primary)" }}
      >
        <Building2
          className="w-5 h-5 mr-2"
          style={{ color: "var(--text-secondary)" }}
        />
        All Companies ({companies.length})
      </h2>

      <div className="space-y-3">
        {companies.map((company) => (
          <div
            key={company.id}
            className="rounded-lg p-5 transition-all hover:bg-gray-50"
            style={{
              background: "#ffffff",
              border: "1px solid var(--border)",
            }}
          >
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-start space-x-4 flex-1">
                <div
                  className="w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0"
                  style={{ background: "var(--accent-bg)" }}
                >
                  <Building2
                    className="w-6 h-6"
                    style={{ color: "var(--accent)" }}
                  />
                </div>
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-2">
                    <h3
                      className="font-semibold text-lg"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {company.name}
                    </h3>
                    <Badge
                      variant={company.status === "active" ? "green" : "gray"}
                      size="sm"
                    >
                      {company.status}
                    </Badge>
                  </div>
                  <p
                    className="text-sm mb-2"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {company.industry}
                  </p>
                  <p
                    className="text-sm"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {company.purpose}
                  </p>
                </div>
              </div>

              <div className="flex items-center space-x-2">
                <button
                  onClick={() => onEdit(company)}
                  className="p-2 rounded-lg transition-colors hover:bg-gray-50"
                  style={{ background: "var(--background)" }}
                >
                  <Edit
                    className="w-4 h-4"
                    style={{ color: "var(--text-secondary)" }}
                  />
                </button>
                <button
                  onClick={() => onDelete(company.id)}
                  className="p-2 rounded-lg transition-colors hover:bg-gray-50"
                  style={{ background: "var(--background)" }}
                >
                  <Trash2
                    className="w-4 h-4"
                    style={{ color: "var(--text-secondary)" }}
                  />
                </button>
              </div>
            </div>

            <div
              className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4"
              style={{ borderTop: "1px solid var(--border)" }}
            >
              <div>
                <p
                  className="text-xs mb-1"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Employees
                </p>
                <p
                  className="font-semibold flex items-center"
                  style={{ color: "var(--text-primary)" }}
                >
                  <Users className="w-4 h-4 mr-1" />
                  {company.employees}
                </p>
              </div>
              <div>
                <p
                  className="text-xs mb-1"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Phone
                </p>
                <p className="text-sm" style={{ color: "var(--text-primary)" }}>
                  {company.phone}
                </p>
              </div>
              <div>
                <p
                  className="text-xs mb-1"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Address
                </p>
                <p
                  className="text-sm truncate"
                  style={{ color: "var(--text-primary)" }}
                >
                  {company.address}
                </p>
              </div>
              <div>
                <p
                  className="text-xs mb-1"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Created
                </p>
                <p
                  className="text-sm flex items-center"
                  style={{ color: "var(--text-primary)" }}
                >
                  <Calendar className="w-4 h-4 mr-1" />
                  {formatDate(company.created_at)}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
