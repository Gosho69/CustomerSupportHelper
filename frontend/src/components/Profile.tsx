"use client";

import { useState, useEffect } from "react";
import {
  Mail,
  Phone,
  Building2,
  Calendar,
  TrendingUp,
  Target,
  Clock,
  Star,
  Activity,
  Edit,
} from "lucide-react";
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";
import { useAuthStore } from "@/store/authStore";
import { callsApi, reportsApi } from "@/lib/api";

export default function Profile() {
  const { user, setAuth } = useAuthStore();
  const [isEditing, setIsEditing] = useState(false);
  const [profilePicture, setProfilePicture] = useState<string | null>(null);
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
    currentStreak: 0,
  });
  const [skillData, setSkillData] = useState<any[]>([]);
  const [monthlyActivity, setMonthlyActivity] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (user) {
      setEditedProfile({
        first_name: user.first_name || "",
        last_name: user.last_name || "",
        email: user.email || "",
        phone: user.phone || "",
      });

      if (user.role !== "admin") {
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

      const calls = callsResponse.data;
      const reports = reportsResponse.data;

      // Calculate stats from calls
      const totalCalls = calls.length;
      const totalDuration = calls.reduce(
        (sum: number, call: any) => sum + (call.duration || 0),
        0,
      );
      const totalHours = Math.round(totalDuration / 3600);

      // Calculate average score from reports
      const scores = reports
        .filter((r: any) => r.metrics?.overall_score)
        .map((r: any) => r.metrics.overall_score);
      const avgScore =
        scores.length > 0
          ? Math.round(
              scores.reduce((a: number, b: number) => a + b, 0) / scores.length,
            )
          : 0;

      setStats({
        totalCalls,
        avgScore,
        totalHours,
        currentStreak: 0, // This would need backend implementation
      });

      // Extract skill data from latest report
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

      // Generate monthly activity from calls
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
    department: "Customer Support",
    role: getRoleLabel(user.role),
    joinDate: "Jan 15, 2024", // This would need to be added to user model or calculated from created_at
    avatar: null,
  };

  const handleProfilePictureChange = (
    e: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setProfilePicture(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSaveProfile = async () => {
    if (!user) return;

    try {
      const { usersApi } = await import("@/lib/api");
      await usersApi.updateUser(user.id, editedProfile);

      // Update auth store with new data
      const updatedUser = { ...user, ...editedProfile };
      setAuth(
        updatedUser,
        localStorage.getItem("access_token") || "",
        localStorage.getItem("refresh_token") || "",
      );

      setIsEditing(false);
      alert("Profile updated successfully!");
    } catch (error) {
      console.error("Failed to update profile:", error);
      alert("Failed to update profile");
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

  const isAdmin = user.role === "admin";

  const cardStyle: React.CSSProperties = {
    background: "#ffffff",
    border: "1px solid var(--border, #e3e8ee)",
    borderRadius: "8px",
  };

  return (
    <div className="space-y-6">
      {/* Header with Profile Card */}
      <div
        className="rounded-lg p-8 text-white"
        style={{ background: "var(--accent)" }}
      >
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between">
          <div className="flex items-center space-x-6 mb-6 md:mb-0">
            <div className="relative">
              {profilePicture ? (
                <img
                  src={profilePicture}
                  alt="Profile"
                  className="w-24 h-24 rounded-full object-cover border-4 border-white/30"
                />
              ) : (
                <div className="w-24 h-24 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center text-4xl font-bold border-4 border-white/30">
                  {profile.firstName[0]}
                  {profile.lastName[0]}
                </div>
              )}
              {isEditing && (
                <label className="absolute bottom-0 right-0 w-8 h-8 rounded-full flex items-center justify-center cursor-pointer transition-opacity bg-white">
                  <Edit
                    className="w-4 h-4"
                    style={{ color: "var(--accent)" }}
                  />
                  <input
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleProfilePictureChange}
                  />
                </label>
              )}
            </div>
            <div>
              {isEditing ? (
                <div className="space-y-2">
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={editedProfile.first_name}
                      onChange={(e) =>
                        setEditedProfile({
                          ...editedProfile,
                          first_name: e.target.value,
                        })
                      }
                      className="px-3 py-1 bg-white/20 border border-white/30 rounded-lg text-white placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-white/50"
                      placeholder="First Name"
                    />
                    <input
                      type="text"
                      value={editedProfile.last_name}
                      onChange={(e) =>
                        setEditedProfile({
                          ...editedProfile,
                          last_name: e.target.value,
                        })
                      }
                      className="px-3 py-1 bg-white/20 border border-white/30 rounded-lg text-white placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-white/50"
                      placeholder="Last Name"
                    />
                  </div>
                  <input
                    type="email"
                    value={editedProfile.email}
                    onChange={(e) =>
                      setEditedProfile({
                        ...editedProfile,
                        email: e.target.value,
                      })
                    }
                    className="w-full px-3 py-1 bg-white/20 border border-white/30 rounded-lg text-white placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-white/50"
                    placeholder="Email"
                  />
                  <input
                    type="tel"
                    value={editedProfile.phone}
                    onChange={(e) =>
                      setEditedProfile({
                        ...editedProfile,
                        phone: e.target.value,
                      })
                    }
                    className="w-full px-3 py-1 bg-white/20 border border-white/30 rounded-lg text-white placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-white/50"
                    placeholder="Phone Number"
                  />
                </div>
              ) : (
                <h1 className="text-3xl font-bold mb-2">
                  {profile.firstName} {profile.lastName}
                </h1>
              )}
              <p className="text-white/80 mb-1">{profile.role}</p>
              <div className="flex items-center space-x-4 text-sm text-white/70">
                <span className="flex items-center">
                  <Building2 className="w-4 h-4 mr-1" />
                  {profile.company}
                </span>
                <span className="flex items-center">
                  <Calendar className="w-4 h-4 mr-1" />
                  Joined {profile.joinDate}
                </span>
              </div>
            </div>
          </div>
          <div className="flex gap-2">
            {isEditing ? (
              <>
                <button
                  onClick={handleSaveProfile}
                  className="px-6 py-3 bg-white rounded-lg font-semibold transition-all flex items-center space-x-2"
                  style={{ color: "var(--accent)" }}
                >
                  <span>Save</span>
                </button>
                <button
                  onClick={handleCancelEdit}
                  className="px-6 py-3 bg-white/20 hover:bg-white/30 backdrop-blur-sm rounded-lg font-semibold transition-all flex items-center space-x-2"
                >
                  <span>Cancel</span>
                </button>
              </>
            ) : (
              <button
                onClick={() => setIsEditing(true)}
                className="px-6 py-3 bg-white/20 hover:bg-white/30 rounded-lg font-semibold transition-all flex items-center space-x-2"
              >
                <Edit className="w-4 h-4" />
                <span>Edit Profile</span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Contact Information */}
      <div className="rounded-lg p-6" style={cardStyle}>
        <h3
          className="text-xl font-bold mb-4"
          style={{ color: "var(--text-primary)" }}
        >
          Contact Information
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label
              className="block text-sm mb-2"
              style={{ color: "var(--text-secondary)" }}
            >
              Email
            </label>
            <div
              className="flex items-center space-x-2 p-3 rounded-lg"
              style={{ background: "var(--background)" }}
            >
              <Mail
                className="w-5 h-5"
                style={{ color: "var(--text-secondary)" }}
              />
              <span style={{ color: "var(--text-primary)" }}>
                {profile.email}
              </span>
            </div>
          </div>
          <div>
            <label
              className="block text-sm mb-2"
              style={{ color: "var(--text-secondary)" }}
            >
              Phone
            </label>
            {isEditing ? (
              <input
                type="tel"
                value={editedProfile.phone}
                onChange={(e) =>
                  setEditedProfile({ ...editedProfile, phone: e.target.value })
                }
                className="w-full px-4 py-3 rounded-lg focus:outline-none focus:ring-2"
                style={{
                  background: "var(--background)",
                  border: "1px solid var(--border)",
                  color: "var(--text-primary)",
                }}
                placeholder="+1 (555) 123-4567"
              />
            ) : (
              <div
                className="flex items-center space-x-2 p-3 rounded-lg"
                style={{ background: "var(--background)" }}
              >
                <Phone
                  className="w-5 h-5"
                  style={{ color: "var(--text-secondary)" }}
                />
                <span style={{ color: "var(--text-primary)" }}>
                  {profile.phone}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Performance Stats - Hidden for Admins */}
      {!isAdmin && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="rounded-lg p-6" style={cardStyle}>
              <div
                className="w-10 h-10 rounded-lg flex items-center justify-center mb-4"
                style={{ background: "var(--accent-bg)" }}
              >
                <Phone className="w-5 h-5" style={{ color: "var(--accent)" }} />
              </div>
              <p
                className="text-sm mb-1"
                style={{ color: "var(--text-secondary)" }}
              >
                Total Calls
              </p>
              <p
                className="text-3xl font-bold"
                style={{ color: "var(--text-primary)" }}
              >
                {stats.totalCalls}
              </p>
            </div>

            <div className="rounded-lg p-6" style={cardStyle}>
              <div
                className="w-10 h-10 rounded-lg flex items-center justify-center mb-4"
                style={{ background: "var(--accent-bg)" }}
              >
                <Star className="w-5 h-5" style={{ color: "var(--accent)" }} />
              </div>
              <p
                className="text-sm mb-1"
                style={{ color: "var(--text-secondary)" }}
              >
                Avg Score
              </p>
              <p
                className="text-3xl font-bold"
                style={{ color: "var(--text-primary)" }}
              >
                {stats.avgScore}
              </p>
            </div>

            <div className="rounded-lg p-6" style={cardStyle}>
              <div
                className="w-10 h-10 rounded-lg flex items-center justify-center mb-4"
                style={{ background: "var(--accent-bg)" }}
              >
                <Clock className="w-5 h-5" style={{ color: "var(--accent)" }} />
              </div>
              <p
                className="text-sm mb-1"
                style={{ color: "var(--text-secondary)" }}
              >
                Total Hours
              </p>
              <p
                className="text-3xl font-bold"
                style={{ color: "var(--text-primary)" }}
              >
                {stats.totalHours}h
              </p>
            </div>

            <div className="rounded-lg p-6" style={cardStyle}>
              <div
                className="w-10 h-10 rounded-lg flex items-center justify-center mb-4"
                style={{ background: "var(--accent-bg)" }}
              >
                <TrendingUp
                  className="w-5 h-5"
                  style={{ color: "var(--accent)" }}
                />
              </div>
              <p
                className="text-sm mb-1"
                style={{ color: "var(--text-secondary)" }}
              >
                Current Streak
              </p>
              <p
                className="text-3xl font-bold"
                style={{ color: "var(--text-primary)" }}
              >
                {stats.currentStreak}
              </p>
            </div>
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Skill Radar */}
            <div className="rounded-lg p-6" style={cardStyle}>
              <h3
                className="text-xl font-bold mb-4 flex items-center"
                style={{ color: "var(--text-primary)" }}
              >
                <Target
                  className="w-5 h-5 mr-2"
                  style={{ color: "var(--accent)" }}
                />
                Skill Assessment
              </h3>
              {skillData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={skillData}>
                    <PolarGrid stroke="#e3e8ee" />
                    <PolarAngleAxis dataKey="skill" stroke="#697386" />
                    <PolarRadiusAxis domain={[0, 100]} stroke="#697386" />
                    <Radar
                      name="Score"
                      dataKey="score"
                      stroke="#635bff"
                      fill="#635bff"
                      fillOpacity={0.6}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              ) : (
                <div
                  className="h-[300px] flex items-center justify-center"
                  style={{ color: "var(--text-secondary)" }}
                >
                  No skill data available yet
                </div>
              )}
            </div>

            {/* Activity Chart */}
            <div className="rounded-lg p-6" style={cardStyle}>
              <h3
                className="text-xl font-bold mb-4 flex items-center"
                style={{ color: "var(--text-primary)" }}
              >
                <Activity
                  className="w-5 h-5 mr-2"
                  style={{ color: "var(--accent)" }}
                />
                Monthly Activity
              </h3>
              {monthlyActivity.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={monthlyActivity}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ee" />
                    <XAxis dataKey="month" stroke="#697386" />
                    <YAxis yAxisId="left" stroke="#697386" />
                    <YAxis
                      yAxisId="right"
                      orientation="right"
                      stroke="#697386"
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#ffffff",
                        border: "1px solid #e3e8ee",
                        borderRadius: "8px",
                        color: "#1a1f36",
                      }}
                    />
                    <Line
                      yAxisId="left"
                      type="monotone"
                      dataKey="calls"
                      stroke="#635bff"
                      strokeWidth={2}
                      name="Calls"
                    />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="score"
                      stroke="#0caf60"
                      strokeWidth={2}
                      name="Score"
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div
                  className="h-[300px] flex items-center justify-center"
                  style={{ color: "var(--text-secondary)" }}
                >
                  No activity data available yet
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
