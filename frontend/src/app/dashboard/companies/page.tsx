"use client";

import { useState, useEffect, useCallback } from "react";
import {
  CompaniesHeader,
  CompaniesStats,
  CompaniesFilters,
  CompaniesTable,
  CompanyModal,
} from "@/components/admin/companies";
import { companiesApi } from "@/lib/api";
import { useToast, ConfirmDialog } from "@/components/ui";

export default function CompaniesPage() {
  const toast = useToast();
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCompany, setSelectedCompany] = useState<any>(null);
  const [showModal, setShowModal] = useState(false);
  const [companies, setCompanies] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [confirmDelete, setConfirmDelete] = useState<{
    open: boolean;
    companyId: number | null;
  }>({
    open: false,
    companyId: null,
  });

  // Fetch companies from backend
  const fetchCompanies = useCallback(async () => {
    try {
      setLoading(true);
      const response = await companiesApi.getAllCompanies();
      setCompanies(response.data || []);
    } catch (error) {
      console.error("Failed to fetch companies:", error);
      setCompanies([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchCompanies();
  }, [fetchCompanies]);

  const filteredCompanies = companies.filter((company) => {
    const matchesSearch =
      company.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      company.industry?.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesSearch;
  });

  const handleAddCompany = () => {
    setSelectedCompany(null);
    setShowModal(true);
  };

  const handleEditCompany = (company: any) => {
    setSelectedCompany(company);
    setShowModal(true);
  };

  const handleDeleteCompany = async (id: number) => {
    setConfirmDelete({ open: true, companyId: id });
  };

  const executeDeleteCompany = async () => {
    const id = confirmDelete.companyId;
    setConfirmDelete({ open: false, companyId: null });
    if (!id) return;
    try {
      await companiesApi.deleteCompany(id);
      toast.success("Company deleted successfully");
      await fetchCompanies();
    } catch (error) {
      console.error("Failed to delete company:", error);
      toast.error("Failed to delete company");
    }
  };

  const handleSaveCompany = async (companyData: any) => {
    try {
      if (selectedCompany) {
        // Edit existing
        await companiesApi.updateCompany(selectedCompany.id, companyData);
      } else {
        // Add new
        await companiesApi.createCompany(companyData);
      }
      setShowModal(false);
      toast.success(
        selectedCompany
          ? "Company updated successfully"
          : "Company created successfully",
      );
      // Refresh the list after save
      await fetchCompanies();
    } catch (error) {
      console.error("Failed to save company:", error);
      toast.error("Failed to save company");
    }
  };

  const stats = {
    total: companies.length,
    totalEmployees: companies.reduce((sum, c) => sum + (c.employees || 0), 0),
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-400">Loading companies...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <CompaniesHeader onAddCompany={handleAddCompany} />

      <CompaniesStats
        total={stats.total}
        totalEmployees={stats.totalEmployees}
      />

      <CompaniesFilters
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
      />

      <CompaniesTable
        companies={filteredCompanies}
        onEdit={handleEditCompany}
        onDelete={handleDeleteCompany}
      />

      {showModal && (
        <CompanyModal
          company={selectedCompany}
          onClose={() => setShowModal(false)}
          onSave={handleSaveCompany}
        />
      )}

      <ConfirmDialog
        open={confirmDelete.open}
        title="Delete Company"
        message="This action cannot be undone. The company and all associated data will be permanently removed."
        confirmLabel="Delete Company"
        variant="danger"
        onConfirm={executeDeleteCompany}
        onCancel={() => setConfirmDelete({ open: false, companyId: null })}
      />
    </div>
  );
}
