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
    <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
      <h2 className="text-xl font-bold text-white mb-6 flex items-center">
        <Building2 className="w-5 h-5 mr-2 text-indigo-400" />
        All Companies ({companies.length})
      </h2>

      <div className="space-y-3">
        {companies.map((company) => (
          <div
            key={company.id}
            className="bg-slate-900/50 rounded-xl p-5 hover:bg-slate-900/70 transition-all"
          >
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-start space-x-4 flex-1">
                <div className="w-12 h-12 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-lg flex items-center justify-center flex-shrink-0">
                  <Building2 className="w-6 h-6 text-white" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-2">
                    <h3 className="text-white font-semibold text-lg">
                      {company.name}
                    </h3>
                    <Badge
                      variant={company.status === "active" ? "green" : "gray"}
                      size="sm"
                    >
                      {company.status}
                    </Badge>
                  </div>
                  <p className="text-gray-400 text-sm mb-2">
                    {company.industry}
                  </p>
                  <p className="text-gray-500 text-sm">{company.purpose}</p>
                </div>
              </div>

              <div className="flex items-center space-x-2">
                <button
                  onClick={() => onEdit(company)}
                  className="p-2 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 rounded-lg transition-colors"
                >
                  <Edit className="w-4 h-4" />
                </button>
                <button
                  onClick={() => onDelete(company.id)}
                  className="p-2 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-lg transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-white/10">
              <div>
                <p className="text-gray-400 text-xs mb-1">Employees</p>
                <p className="text-white font-semibold flex items-center">
                  <Users className="w-4 h-4 mr-1" />
                  {company.employees}
                </p>
              </div>
              <div>
                <p className="text-gray-400 text-xs mb-1">Phone</p>
                <p className="text-white text-sm">{company.phone}</p>
              </div>
              <div>
                <p className="text-gray-400 text-xs mb-1">Address</p>
                <p className="text-white text-sm truncate">{company.address}</p>
              </div>
              <div>
                <p className="text-gray-400 text-xs mb-1">Created</p>
                <p className="text-white text-sm flex items-center">
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
