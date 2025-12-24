"use client";

import { useState } from "react";
import {
  UsersHeader,
  UsersStats,
  UsersFilters,
  UsersTable,
  UserModal,
} from "@/components/admin/users";

export default function UsersPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [filterRole, setFilterRole] = useState<string>("all");
  const [selectedUser, setSelectedUser] = useState<any>(null);
  const [showModal, setShowModal] = useState(false);

  const [users, setUsers] = useState([
    {
      id: 1,
      username: "admin1",
      email: "admin@techsolutions.com",
      first_name: "John",
      last_name: "Admin",
      role: "admin" as const,
      company: "Tech Solutions Inc",
      is_active: true,
      created_at: "2024-01-15",
    },
    {
      id: 2,
      username: "head1",
      email: "sarah.head@techsolutions.com",
      first_name: "Sarah",
      last_name: "Johnson",
      role: "head_of_department" as const,
      company: "Tech Solutions Inc",
      reporting_to: undefined,
      is_active: true,
      created_at: "2024-01-20",
    },
    {
      id: 3,
      username: "agent1",
      email: "mike.agent@techsolutions.com",
      first_name: "Mike",
      last_name: "Davis",
      role: "agent" as const,
      company: "Tech Solutions Inc",
      reporting_to: "Sarah Johnson",
      is_active: true,
      created_at: "2024-02-01",
    },
    {
      id: 4,
      username: "agent2",
      email: "emma.agent@retail.com",
      first_name: "Emma",
      last_name: "Wilson",
      role: "agent" as const,
      company: "Global Retail Corp",
      reporting_to: "Sarah Johnson",
      is_active: false,
      created_at: "2024-02-15",
    },
  ]);

  const filteredUsers = users.filter((user) => {
    const matchesSearch =
      user.first_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      user.last_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      user.email.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesRole = filterRole === "all" || user.role === filterRole;
    return matchesSearch && matchesRole;
  });

  const handleAddUser = () => {
    setSelectedUser(null);
    setShowModal(true);
  };

  const handleEditUser = (user: any) => {
    setSelectedUser(user);
    setShowModal(true);
  };

  const handleDeleteUser = (id: number) => {
    if (confirm("Are you sure you want to delete this user?")) {
      setUsers(users.filter((u) => u.id !== id));
    }
  };

  const handleSaveUser = (userData: any) => {
    if (selectedUser) {
      setUsers(
        users.map((u) => (u.id === selectedUser.id ? { ...u, ...userData } : u))
      );
    } else {
      setUsers([
        ...users,
        {
          ...userData,
          id: users.length + 1,
          created_at: new Date().toISOString(),
        },
      ]);
    }
    setShowModal(false);
  };

  const stats = {
    total: users.length,
    admins: users.filter((u) => u.role === "admin").length,
    heads: users.filter((u) => u.role === "head_of_department").length,
    agents: users.filter((u) => u.role === "agent").length,
    active: users.filter((u) => u.is_active).length,
  };

  return (
    <div className="space-y-6">
      <UsersHeader onAddUser={handleAddUser} />

      <UsersStats
        total={stats.total}
        admins={stats.admins}
        heads={stats.heads}
        agents={stats.agents}
        active={stats.active}
      />

      <UsersFilters
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
        filterRole={filterRole}
        setFilterRole={setFilterRole}
      />

      <UsersTable
        users={filteredUsers}
        onEdit={handleEditUser}
        onDelete={handleDeleteUser}
      />

      {showModal && (
        <UserModal
          user={selectedUser}
          onClose={() => setShowModal(false)}
          onSave={handleSaveUser}
        />
      )}
    </div>
  );
}
