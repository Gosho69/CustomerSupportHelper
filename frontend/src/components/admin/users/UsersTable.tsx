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
  currentUserId?: number;
  onEdit: (user: User) => void;
  onDelete: (id: number) => void;
}

export default function UsersTable({
  users,
  currentUserId,
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
        <Users
          className="w-5 h-5 mr-2"
          style={{ color: "var(--text-secondary)" }}
        />
        All Users ({users.length})
      </h2>

      <div className="space-y-3">
        {users.map((user) => {
          const roleBadge = getRoleBadge(user.role);
          return (
            <div
              key={user.id}
              className="rounded-lg p-5 transition-all hover:bg-gray-50"
              style={{
                background: "#ffffff",
                border: "1px solid var(--border)",
              }}
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-start space-x-4 flex-1">
                  <div
                    className="w-12 h-12 rounded-full flex items-center justify-center flex-shrink-0"
                    style={{ background: "var(--accent-bg)" }}
                  >
                    <span
                      className="font-semibold"
                      style={{ color: "var(--accent)" }}
                    >
                      {getInitials(user.first_name, user.last_name)}
                    </span>
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <h3
                        className="font-semibold text-lg"
                        style={{ color: "var(--text-primary)" }}
                      >
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
                    <p
                      className="text-sm mb-1"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      @{user.username}
                    </p>
                    <p
                      className="text-sm"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {user.email}
                    </p>
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => onEdit(user)}
                    className="p-2 rounded-lg transition-colors hover:bg-gray-50"
                    style={{ background: "var(--background)" }}
                  >
                    <Edit
                      className="w-4 h-4"
                      style={{ color: "var(--text-secondary)" }}
                    />
                  </button>
                  <button
                    onClick={() => onDelete(user.id)}
                    disabled={currentUserId === user.id}
                    className={`p-2 rounded-lg transition-colors hover:bg-gray-50 ${
                      currentUserId === user.id
                        ? "bg-gray-100 cursor-not-allowed opacity-40"
                        : ""
                    }`}
                    title={
                      currentUserId === user.id
                        ? "You cannot delete your own account"
                        : "Delete user"
                    }
                    style={
                      currentUserId === user.id
                        ? {}
                        : { background: "var(--background)" }
                    }
                  >
                    <Trash2
                      className="w-4 h-4"
                      style={{ color: "var(--text-secondary)" }}
                    />
                  </button>
                </div>
              </div>

              <div
                className="grid grid-cols-2 md:grid-cols-3 gap-4 pt-4"
                style={{ borderTop: "1px solid var(--border)" }}
              >
                <div>
                  <p
                    className="text-xs mb-1"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Company
                  </p>
                  <p
                    className="text-sm flex items-center"
                    style={{ color: "var(--text-primary)" }}
                  >
                    <Building2 className="w-4 h-4 mr-1" />
                    {user.company}
                  </p>
                </div>
                {user.reporting_to && (
                  <div>
                    <p
                      className="text-xs mb-1"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      Reports To
                    </p>
                    <p
                      className="text-sm flex items-center"
                      style={{ color: "var(--text-primary)" }}
                    >
                      <UserCheck className="w-4 h-4 mr-1" />
                      {user.reporting_to}
                    </p>
                  </div>
                )}
                <div>
                  <p
                    className="text-xs mb-1"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Joined
                  </p>
                  <p
                    className="text-sm flex items-center"
                    style={{ color: "var(--text-primary)" }}
                  >
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
