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

export default function CompaniesPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCompany, setSelectedCompany] = useState<any>(null);
  const [showModal, setShowModal] = useState(false);
  const [companies, setCompanies] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

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
    if (confirm("Are you sure you want to delete this company?")) {
      try {
        await companiesApi.deleteCompany(id);
        // Refresh the list after deletion
        await fetchCompanies();
      } catch (error) {
        console.error("Failed to delete company:", error);
        alert("Failed to delete company");
      }
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
      // Refresh the list after save
      await fetchCompanies();
    } catch (error) {
      console.error("Failed to save company:", error);
      alert("Failed to save company");
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
    </div>
  );
}
