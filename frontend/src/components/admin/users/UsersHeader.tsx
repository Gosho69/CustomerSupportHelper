import { UserPlus } from "lucide-react";

interface UsersHeaderProps {
  onAddUser: () => void;
}

export default function UsersHeader({ onAddUser }: UsersHeaderProps) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <h1 className="text-3xl font-bold text-white">Users</h1>
        <p className="text-gray-400 mt-1">
          Manage all users and their roles in the system
        </p>
      </div>
      <button
        onClick={onAddUser}
        className="px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white font-semibold rounded-xl transition-all shadow-lg shadow-indigo-500/30 flex items-center space-x-2"
      >
        <UserPlus className="w-5 h-5" />
        <span>Add User</span>
      </button>
    </div>
  );
}
