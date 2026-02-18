"use client";

import { useState, useEffect } from "react";
import { useAuthStore } from "@/store/authStore";
import { callsApi, reportsApi } from "@/lib/api";
import { useToast } from "@/components/ui";
import ProfileHeader from "./profile/ProfileHeader";
import ContactInfo from "./profile/ContactInfo";
import PerformanceStats from "./profile/PerformanceStats";

export default function Profile() {
  const { user, setAuth } = useAuthStore();
  const toast = useToast();
  const [isEditing, setIsEditing] = useState(false);
  const [editedProfile, setEditedProfile] = useState({
    first_name: "",
    last_name: "",
    email: "",
    phone: "",
  });
  const [stats, setStats] = useState({
    totalCalls: 0,
    avgScore: 0,
    totalHours: 0,
  });
  const [skillData, setSkillData] = useState<any[]>([]);
  const [monthlyActivity, setMonthlyActivity] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (user) {
      setEditedProfile({
        first_name: user.first_name || "",
        last_name: user.last_name || "",
        email: user.email || "",
        phone: user.phone || "",
      });

      if (user.role === "agent") {
        fetchProfileData();
      } else {
        setLoading(false);
      }
    }
  }, [user]);

  const fetchProfileData = async () => {
    try {
      const [callsResponse, reportsResponse] = await Promise.all([
        callsApi.getMyCalls(),
        reportsApi.getMyReports(),
      ]);

      const calls = callsResponse.data || [];
      const rawReports = reportsResponse.data;
      const reports = Array.isArray(rawReports)
        ? rawReports
        : rawReports?.reports || [];

      const totalCalls = calls.length;
      const totalDuration = calls.reduce(
        (sum: number, call: any) => sum + (call.duration || 0),
        0,
      );
      const totalHours = Math.round(totalDuration / 3600);

      const scores = reports
        .filter((r: any) => r.metrics?.overall_score)
        .map((r: any) => r.metrics.overall_score);
      const avgScore =
        scores.length > 0
          ? Math.round(
              scores.reduce((a: number, b: number) => a + b, 0) / scores.length,
            )
          : 0;

      setStats({ totalCalls, avgScore, totalHours });

      if (reports.length > 0 && reports[0].metrics) {
        const metrics = reports[0].metrics;
        setSkillData([
          {
            skill: "Empathy",
            score: Math.round((metrics.empathy_score || 0) * 100),
          },
          {
            skill: "Communication",
            score: Math.round((metrics.communication_score || 0) * 100),
          },
          {
            skill: "Problem Solving",
            score: Math.round((metrics.problem_solving_score || 0) * 100),
          },
          {
            skill: "Product Knowledge",
            score: Math.round((metrics.product_knowledge || 0) * 100),
          },
          {
            skill: "Time Management",
            score: Math.round((metrics.efficiency_score || 0) * 100),
          },
          {
            skill: "Active Listening",
            score: Math.round((metrics.active_listening || 0) * 100),
          },
        ]);
      }

      const monthlyData: any = {};
      calls.forEach((call: any) => {
        const date = new Date(call.created_at);
        const monthKey = date.toLocaleDateString("en-US", { month: "short" });
        if (!monthlyData[monthKey]) {
          monthlyData[monthKey] = {
            month: monthKey,
            calls: 0,
            score: 0,
            scoreCount: 0,
          };
        }
        monthlyData[monthKey].calls++;
      });

      reports.forEach((report: any) => {
        const date = new Date(report.created_at);
        const monthKey = date.toLocaleDateString("en-US", { month: "short" });
        if (monthlyData[monthKey] && report.metrics?.overall_score) {
          monthlyData[monthKey].score += report.metrics.overall_score;
          monthlyData[monthKey].scoreCount++;
        }
      });

      const activityData = Object.values(monthlyData).map((data: any) => ({
        month: data.month,
        calls: data.calls,
        score:
          data.scoreCount > 0 ? Math.round(data.score / data.scoreCount) : 0,
      }));

      setMonthlyActivity(activityData.slice(-6));
      setLoading(false);
    } catch (error) {
      console.error("Error fetching profile data:", error);
      setLoading(false);
    }
  };

  const getRoleLabel = (role: string) => {
    switch (role) {
      case "agent":
        return "Support Agent";
      case "head_of_department":
        return "Head of Department";
      case "admin":
        return "Administrator";
      default:
        return "Support Agent";
    }
  };

  if (!user) {
    return <div style={{ color: "var(--text-primary)" }}>Loading...</div>;
  }

  const profile = {
    firstName: user.first_name || "User",
    lastName: user.last_name || "",
    email: user.email,
    phone: user.phone || "Not set",
    company: user.company_name || "N/A",
    role: getRoleLabel(user.role),
    joinDate: "Jan 15, 2024",
  };

  const handleSaveProfile = async () => {
    if (!user) return;
    setSaving(true);
    try {
      const { api } = await import("@/lib/api");
      await api.patch("/users/me/", editedProfile);
      const updatedUser = { ...user, ...editedProfile };
      setAuth(
        updatedUser,
        localStorage.getItem("access_token") || "",
        localStorage.getItem("refresh_token") || "",
      );
      setIsEditing(false);
      toast.success("Profile updated successfully!");
    } catch (error) {
      console.error("Failed to update profile:", error);
      toast.error("Failed to update profile");
    } finally {
      setSaving(false);
    }
  };

  const handleCancelEdit = () => {
    setEditedProfile({
      first_name: user?.first_name || "",
      last_name: user?.last_name || "",
      email: user?.email || "",
      phone: user?.phone || "",
    });
    setIsEditing(false);
  };

  const isAgent = user.role === "agent";

  return (
    <div className="space-y-6">
      <ProfileHeader
        firstName={profile.firstName}
        lastName={profile.lastName}
        role={profile.role}
        company={profile.company}
        joinDate={profile.joinDate}
        isEditing={isEditing}
        saving={saving}
        editedProfile={editedProfile}
        setEditedProfile={setEditedProfile}
        onSave={handleSaveProfile}
        onCancel={handleCancelEdit}
        onStartEdit={() => setIsEditing(true)}
      />

      <ContactInfo
        email={profile.email}
        phone={profile.phone}
        isEditing={isEditing}
        editedPhone={editedProfile.phone}
        onPhoneChange={(phone) => setEditedProfile({ ...editedProfile, phone })}
      />

      {isAgent && (
        <PerformanceStats
          stats={stats}
          skillData={skillData}
          monthlyActivity={monthlyActivity}
        />
      )}
    </div>
  );
}
