"use client";

import { useState, useEffect, useCallback } from "react";
import {
  UsersHeader,
  UsersStats,
  UsersFilters,
  UsersTable,
  UserModal,
} from "@/components/admin/users";
import { usersApi } from "@/lib/api";
import { useAuthStore } from "@/store/authStore";
import { useToast, ConfirmDialog } from "@/components/ui";

export default function UsersPage() {
  const { user: currentUser } = useAuthStore();
  const toast = useToast();
  const [searchQuery, setSearchQuery] = useState("");
  const [filterRole, setFilterRole] = useState<string>("all");
  const [selectedUser, setSelectedUser] = useState<any>(null);
  const [showModal, setShowModal] = useState(false);
  const [users, setUsers] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [confirmDelete, setConfirmDelete] = useState<{
    open: boolean;
    userId: number | null;
  }>({
    open: false,
    userId: null,
  });

  // Fetch users from backend
  const fetchUsers = useCallback(async () => {
    try {
      setLoading(true);
      const response = await usersApi.getAllUsers(
        filterRole === "all" ? undefined : filterRole,
      );
      setUsers(response.data || []);
    } catch (error) {
      console.error("Failed to fetch users:", error);
      setUsers([]);
    } finally {
      setLoading(false);
    }
  }, [filterRole]);

  useEffect(() => {
    fetchUsers();
  }, [fetchUsers]);

  const filteredUsers = users.filter((user) => {
    // Exclude the current logged-in user from the list
    if (currentUser && user.id === currentUser.id) {
      return false;
    }

    const matchesSearch =
      user.first_name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      user.last_name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      user.email?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      user.username?.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesSearch;
  });

  const handleAddUser = () => {
    setSelectedUser(null);
    setShowModal(true);
  };

  const handleEditUser = (user: any) => {
    setSelectedUser(user);
    setShowModal(true);
  };

  const handleDeleteUser = async (id: number) => {
    // Prevent self-deletion
    if (currentUser && id === currentUser.id) {
      toast.warning("You cannot delete your own account!");
      return;
    }
    setConfirmDelete({ open: true, userId: id });
  };

  const executeDeleteUser = async () => {
    const id = confirmDelete.userId;
    setConfirmDelete({ open: false, userId: null });
    if (!id) return;
    try {
      await usersApi.deleteUser(id);
      toast.success("User deleted successfully");
      await fetchUsers();
    } catch (error) {
      console.error("Failed to delete user:", error);
      toast.error("Failed to delete user");
    }
  };

  const handleSaveUser = async (userData: any) => {
    try {
      // Convert string values to integers for foreign keys
      const processedData = {
        ...userData,
        company: userData.company ? parseInt(userData.company) : null,
        reporting_to: userData.reporting_to
          ? parseInt(userData.reporting_to)
          : null,
      };

      if (selectedUser) {
        // Edit existing
        await usersApi.updateUser(selectedUser.id, processedData);
      } else {
        // Add new - use appropriate endpoint based on role
        if (userData.role === "agent") {
          await usersApi.createAgent(processedData);
        } else if (userData.role === "head_of_department") {
          await usersApi.createHead(processedData);
        } else {
          throw new Error("Cannot create admin users from this interface");
        }
      }
      setShowModal(false);
      toast.success(
        selectedUser
          ? "User updated successfully"
          : "User created successfully",
      );
      // Refresh the list after save
      await fetchUsers();
    } catch (error) {
      console.error("Failed to save user:", error);
      toast.error("Failed to save user");
    }
  };

  const stats = {
    total: users.length,
    admins: users.filter((u) => u.role === "admin").length,
    heads: users.filter((u) => u.role === "head_of_department").length,
    agents: users.filter((u) => u.role === "agent").length,
    active: users.filter((u) => u.is_active).length,
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-400">Loading users...</div>
      </div>
    );
  }

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
        currentUserId={currentUser?.id}
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

      <ConfirmDialog
        open={confirmDelete.open}
        title="Delete User"
        message="This action cannot be undone. The user will be permanently removed from the system."
        confirmLabel="Delete User"
        variant="danger"
        onConfirm={executeDeleteUser}
        onCancel={() => setConfirmDelete({ open: false, userId: null })}
      />
    </div>
  );
}
