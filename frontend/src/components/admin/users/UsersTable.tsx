import {
  Users,
  Edit,
  Trash2,
  Building2,
  Calendar,
  UserCheck,
} from "lucide-react";
import { Badge } from "@/components/ui";

interface User {
  id: number;
  username: string;
  email: string;
  first_name: string;
  last_name: string;
  role: "admin" | "head_of_department" | "agent";
  company: string;
  reporting_to?: string;
  is_active: boolean;
  created_at: string;
}

interface UsersTableProps {
  users: User[];
  onEdit: (user: User) => void;
  onDelete: (id: number) => void;
}

export default function UsersTable({
  users,
  onEdit,
  onDelete,
}: UsersTableProps) {
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  };

  const getRoleBadge = (role: string) => {
    const roleMap: Record<
      string,
      { variant: "purple" | "blue" | "gray"; label: string }
    > = {
      admin: { variant: "purple", label: "Admin" },
      head_of_department: { variant: "blue", label: "Head" },
      agent: { variant: "gray", label: "Agent" },
    };
    return roleMap[role] || { variant: "gray", label: role };
  };

  const getInitials = (firstName: string, lastName: string) => {
    return `${firstName.charAt(0)}${lastName.charAt(0)}`.toUpperCase();
  };

  return (
    <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
      <h2 className="text-xl font-bold text-white mb-6 flex items-center">
        <Users className="w-5 h-5 mr-2 text-indigo-400" />
        All Users ({users.length})
      </h2>

      <div className="space-y-3">
        {users.map((user) => {
          const roleBadge = getRoleBadge(user.role);
          return (
            <div
              key={user.id}
              className="bg-slate-900/50 rounded-xl p-5 hover:bg-slate-900/70 transition-all"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-start space-x-4 flex-1">
                  <div className="w-12 h-12 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-white font-semibold">
                      {getInitials(user.first_name, user.last_name)}
                    </span>
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <h3 className="text-white font-semibold text-lg">
                        {user.first_name} {user.last_name}
                      </h3>
                      <Badge variant={roleBadge.variant} size="sm">
                        {roleBadge.label}
                      </Badge>
                      {user.is_active && (
                        <Badge variant="green" size="sm">
                          Active
                        </Badge>
                      )}
                    </div>
                    <p className="text-gray-400 text-sm mb-1">
                      @{user.username}
                    </p>
                    <p className="text-gray-500 text-sm">{user.email}</p>
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => onEdit(user)}
                    className="p-2 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 rounded-lg transition-colors"
                  >
                    <Edit className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => onDelete(user.id)}
                    className="p-2 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-lg transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 pt-4 border-t border-white/10">
                <div>
                  <p className="text-gray-400 text-xs mb-1">Company</p>
                  <p className="text-white text-sm flex items-center">
                    <Building2 className="w-4 h-4 mr-1" />
                    {user.company}
                  </p>
                </div>
                {user.reporting_to && (
                  <div>
                    <p className="text-gray-400 text-xs mb-1">Reports To</p>
                    <p className="text-white text-sm flex items-center">
                      <UserCheck className="w-4 h-4 mr-1" />
                      {user.reporting_to}
                    </p>
                  </div>
                )}
                <div>
                  <p className="text-gray-400 text-xs mb-1">Joined</p>
                  <p className="text-white text-sm flex items-center">
                    <Calendar className="w-4 h-4 mr-1" />
                    {formatDate(user.created_at)}
                  </p>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
