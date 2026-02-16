"use client";

import { useState, useEffect } from "react";
import { X } from "lucide-react";
import { createPortal } from "react-dom";
import { companiesApi, usersApi } from "@/lib/api";
import { CustomSelect } from "@/components/ui";

interface UserModalProps {
  user: any;
  onClose: () => void;
  onSave: (data: any) => void;
}

export default function UserModal({ user, onClose, onSave }: UserModalProps) {
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    first_name: "",
    last_name: "",
    phone: "",
    role: "agent",
    company: "",
    reporting_to: "",
    is_active: true,
  });
  const [companies, setCompanies] = useState<any[]>([]);
  const [heads, setHeads] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDropdownData();
  }, []);

  useEffect(() => {
    if (user) {
      setFormData({
        username: user.username || "",
        email: user.email || "",
        password: "", // Don't populate password for edit
        first_name: user.first_name || "",
        last_name: user.last_name || "",
        phone: user.phone || "",
        role: user.role || "agent",
        company: user.company || "",
        reporting_to: user.reporting_to || "",
        is_active: user.is_active ?? true,
      });
    }
  }, [user]);

  const fetchDropdownData = async () => {
    try {
      const [companiesResponse, headsResponse] = await Promise.all([
        companiesApi.getAllCompanies(),
        usersApi.getAllUsers("head_of_department"),
      ]);
      setCompanies(companiesResponse.data || []);
      setHeads(headsResponse.data || []);
    } catch (error) {
      console.error("Failed to fetch dropdown data:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave(formData);
  };

  // Prepare options for dropdowns
  const companyOptions = companies.map((company) => ({
    value: company.id.toString(),
    label: company.name,
    subtitle: `${company.employees || 0} employees`,
  }));

  const headOptions = heads.map((head) => ({
    value: head.id.toString(),
    label: `${head.first_name} ${head.last_name}`,
    subtitle: head.email,
  }));

  const roleOptions = [
    { value: "admin", label: "Admin", subtitle: "Full system access" },
    {
      value: "head_of_department",
      label: "Head of Department",
      subtitle: "Manages teams and agents",
    },
    { value: "agent", label: "Agent", subtitle: "Handles customer calls" },
  ];

  const modalContent = (
    <div className="fixed inset-0 z-[99999] flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/20" onClick={onClose} />
      <div
        className="relative rounded-lg p-6 w-full max-w-2xl shadow-2xl max-h-[90vh] overflow-y-auto"
        style={{ background: "#ffffff", border: "1px solid var(--border)" }}
      >
        <div className="flex items-center justify-between mb-6">
          <h2
            className="text-2xl font-bold"
            style={{ color: "var(--text-primary)" }}
          >
            {user ? "Edit User" : "Add New User"}
          </h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-50 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" style={{ color: "var(--text-secondary)" }} />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label
                className="block text-sm font-medium mb-2"
                style={{ color: "var(--text-secondary)" }}
              >
                First Name
              </label>
              <input
                type="text"
                value={formData.first_name}
                onChange={(e) =>
                  setFormData({ ...formData, first_name: e.target.value })
                }
                className="w-full px-4 py-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                style={{
                  background: "#ffffff",
                  border: "1px solid var(--border)",
                  color: "var(--text-primary)",
                }}
                placeholder="John"
              />
            </div>

            <div>
              <label
                className="block text-sm font-medium mb-2"
                style={{ color: "var(--text-secondary)" }}
              >
                Last Name
              </label>
              <input
                type="text"
                value={formData.last_name}
                onChange={(e) =>
                  setFormData({ ...formData, last_name: e.target.value })
                }
                className="w-full px-4 py-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                style={{
                  background: "#ffffff",
                  border: "1px solid var(--border)",
                  color: "var(--text-primary)",
                }}
                placeholder="Doe"
              />
            </div>
          </div>

          <div>
            <label
              className="block text-sm font-medium mb-2"
              style={{ color: "var(--text-secondary)" }}
            >
              Username *
            </label>
            <input
              type="text"
              required
              value={formData.username}
              onChange={(e) =>
                setFormData({ ...formData, username: e.target.value })
              }
              className="w-full px-4 py-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
              style={{
                background: "#ffffff",
                border: "1px solid var(--border)",
                color: "var(--text-primary)",
              }}
              placeholder="johndoe"
            />
          </div>

          <div>
            <label
              className="block text-sm font-medium mb-2"
              style={{ color: "var(--text-secondary)" }}
            >
              Email *
            </label>
            <input
              type="email"
              required
              value={formData.email}
              onChange={(e) =>
                setFormData({ ...formData, email: e.target.value })
              }
              className="w-full px-4 py-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
              style={{
                background: "#ffffff",
                border: "1px solid var(--border)",
                color: "var(--text-primary)",
              }}
              placeholder="john@company.com"
            />
          </div>

          {!user && (
            <div>
              <label
                className="block text-sm font-medium mb-2"
                style={{ color: "var(--text-secondary)" }}
              >
                Password *
              </label>
              <input
                type="password"
                required
                value={formData.password}
                onChange={(e) =>
                  setFormData({ ...formData, password: e.target.value })
                }
                className="w-full px-4 py-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                style={{
                  background: "#ffffff",
                  border: "1px solid var(--border)",
                  color: "var(--text-primary)",
                }}
                placeholder="Minimum 8 characters"
                minLength={8}
              />
            </div>
          )}

          <div>
            <label
              className="block text-sm font-medium mb-2"
              style={{ color: "var(--text-secondary)" }}
            >
              Phone
            </label>
            <input
              type="tel"
              value={formData.phone}
              onChange={(e) =>
                setFormData({ ...formData, phone: e.target.value })
              }
              className="w-full px-4 py-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
              style={{
                background: "#ffffff",
                border: "1px solid var(--border)",
                color: "var(--text-primary)",
              }}
              placeholder="+1 (555) 123-4567"
            />
          </div>

          <div
            className={`grid ${formData.role === "admin" ? "grid-cols-1" : "grid-cols-2"} gap-4`}
          >
            <CustomSelect
              label="Role"
              required
              options={roleOptions}
              value={formData.role}
              onChange={(value) => setFormData({ ...formData, role: value })}
              placeholder="Select a role"
              searchable={false}
            />

            {formData.role !== "admin" && (
              <CustomSelect
                label="Company"
                required
                options={companyOptions}
                value={formData.company}
                onChange={(value) =>
                  setFormData({ ...formData, company: value })
                }
                placeholder="Select a company"
              />
            )}
          </div>

          {formData.role === "agent" && (
            <CustomSelect
              label="Reports To (Head of Department)"
              options={headOptions}
              value={formData.reporting_to}
              onChange={(value) =>
                setFormData({ ...formData, reporting_to: value })
              }
              placeholder="Select a manager (optional)"
            />
          )}

          <div className="flex items-center gap-3 py-2">
            <input
              type="checkbox"
              id="is_active"
              checked={formData.is_active}
              onChange={(e) =>
                setFormData({ ...formData, is_active: e.target.checked })
              }
              className="styled-checkbox"
            />
            <label
              htmlFor="is_active"
              className="text-sm font-medium cursor-pointer select-none"
              style={{ color: "var(--text-secondary)" }}
            >
              Active User
            </label>
          </div>

          <div className="flex gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-6 py-3 hover:bg-gray-50 font-semibold rounded-lg transition-colors"
              style={{
                background: "#ffffff",
                border: "1px solid var(--border)",
                color: "var(--text-primary)",
              }}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="flex-1 px-6 py-3 font-semibold rounded-lg transition-all shadow-lg flex items-center justify-center"
              style={{ background: "var(--accent-bg)", color: "var(--accent)" }}
            >
              {user ? "Save Changes" : "Add User"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );

  return typeof document !== "undefined"
    ? createPortal(modalContent, document.body)
    : null;
}
