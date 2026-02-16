import { UserPlus } from "lucide-react";

interface UsersHeaderProps {
  onAddUser: () => void;
}

export default function UsersHeader({ onAddUser }: UsersHeaderProps) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <h1
          className="text-3xl font-bold"
          style={{ color: "var(--text-primary)" }}
        >
          Users
        </h1>
        <p className="mt-1" style={{ color: "var(--text-secondary)" }}>
          Manage all users and their roles in the system
        </p>
      </div>
      <button
        onClick={onAddUser}
        className="px-6 py-3 font-semibold rounded-lg transition-all flex items-center space-x-2"
        style={{ background: "var(--accent-bg)", color: "var(--accent)" }}
      >
        <UserPlus className="w-5 h-5" />
        <span>Add User</span>
      </button>
    </div>
  );
}
