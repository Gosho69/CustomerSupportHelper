"use client";

import { useState } from "react";
import {
  CompaniesHeader,
  CompaniesStats,
  CompaniesFilters,
  CompaniesTable,
  CompanyModal,
} from "@/components/admin/companies";

export default function CompaniesPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [selectedCompany, setSelectedCompany] = useState<any>(null);
  const [showModal, setShowModal] = useState(false);

  const [companies, setCompanies] = useState([
    {
      id: 1,
      name: "Tech Solutions Inc",
      industry: "Technology",
      address: "123 Tech Street, San Francisco, CA",
      phone: "+1 (555) 123-4567",
      purpose: "Providing innovative tech solutions",
      employees: 15,
      created_at: "2024-01-15",
      status: "active" as const,
    },
    {
      id: 2,
      name: "Global Retail Corp",
      industry: "Retail",
      address: "456 Commerce Ave, New York, NY",
      phone: "+1 (555) 234-5678",
      purpose: "Leading retail operations worldwide",
      employees: 23,
      created_at: "2024-02-20",
      status: "active" as const,
    },
    {
      id: 3,
      name: "Finance Plus",
      industry: "Finance",
      address: "789 Money Lane, Chicago, IL",
      phone: "+1 (555) 345-6789",
      purpose: "Financial services and consulting",
      employees: 12,
      created_at: "2024-03-10",
      status: "active" as const,
    },
  ]);

  const filteredCompanies = companies.filter((company) => {
    const matchesSearch =
      company.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      company.industry.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus =
      filterStatus === "all" || company.status === filterStatus;
    return matchesSearch && matchesStatus;
  });

  const handleAddCompany = () => {
    setSelectedCompany(null);
    setShowModal(true);
  };

  const handleEditCompany = (company: any) => {
    setSelectedCompany(company);
    setShowModal(true);
  };

  const handleDeleteCompany = (id: number) => {
    if (confirm("Are you sure you want to delete this company?")) {
      setCompanies(companies.filter((c) => c.id !== id));
    }
  };

  const handleSaveCompany = (companyData: any) => {
    if (selectedCompany) {
      // Edit existing
      setCompanies(
        companies.map((c) =>
          c.id === selectedCompany.id ? { ...c, ...companyData } : c
        )
      );
    } else {
      // Add new
      setCompanies([
        ...companies,
        {
          ...companyData,
          id: companies.length + 1,
          created_at: new Date().toISOString(),
        },
      ]);
    }
    setShowModal(false);
  };

  const stats = {
    total: companies.length,
    active: companies.filter((c) => c.status === "active").length,
    totalEmployees: companies.reduce((sum, c) => sum + c.employees, 0),
  };

  return (
    <div className="space-y-6">
      <CompaniesHeader onAddCompany={handleAddCompany} />

      <CompaniesStats
        total={stats.total}
        active={stats.active}
        totalEmployees={stats.totalEmployees}
      />

      <CompaniesFilters
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
        filterStatus={filterStatus}
        setFilterStatus={setFilterStatus}
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
