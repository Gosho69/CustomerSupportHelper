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

export default function Profile() {
  const [isEditing, setIsEditing] = useState(false);
  const [profilePicture, setProfilePicture] = useState<string | null>(null);
  const [userRole, setUserRole] = useState<
    "agent" | "head_of_department" | "admin"
  >("agent");

  useEffect(() => {
    const storedRole = localStorage.getItem("demo_role");
    if (
      storedRole === "head_of_department" ||
      storedRole === "admin" ||
      storedRole === "agent"
    ) {
      setUserRole(storedRole as "agent" | "head_of_department" | "admin");
    }
  }, []);

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

  const profile = {
    firstName: "John",
    lastName: "Smith",
    email: "john.smith@company.com",
    phone: "+1 (555) 123-4567",
    company: "Tech Solutions Inc",
    department: "Customer Support",
    role: getRoleLabel(userRole),
    joinDate: "Jan 15, 2024",
    avatar: null,
  };

  const handleProfilePictureChange = (
    e: React.ChangeEvent<HTMLInputElement>
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

  const stats = {
    totalCalls: 342,
    avgScore: 87,
    totalHours: 156,
    bestStreak: 15,
    currentStreak: 8,
  };

  const skillData = [
    { skill: "Empathy", score: 92 },
    { skill: "Communication", score: 88 },
    { skill: "Problem Solving", score: 85 },
    { skill: "Product Knowledge", score: 79 },
    { skill: "Time Management", score: 83 },
    { skill: "Active Listening", score: 90 },
  ];

  const monthlyActivity = [
    { month: "Jul", calls: 52, score: 84 },
    { month: "Aug", calls: 58, score: 85 },
    { month: "Sep", calls: 61, score: 86 },
    { month: "Oct", calls: 67, score: 87 },
    { month: "Nov", calls: 64, score: 86 },
    { month: "Dec", calls: 40, score: 88 },
  ];

  return (
    <div className="space-y-6">
      {/* Header with Profile Card */}
      <div className="bg-gradient-to-r from-indigo-600 to-blue-600 rounded-2xl p-8 text-white">
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
                <label className="absolute bottom-0 right-0 w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center cursor-pointer hover:bg-blue-600 transition-colors">
                  <Edit className="w-4 h-4 text-white" />
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
              <h1 className="text-3xl font-bold mb-2">
                {profile.firstName} {profile.lastName}
              </h1>
              <p className="text-indigo-100 mb-1">{profile.role}</p>
              <div className="flex items-center space-x-4 text-sm text-indigo-200">
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
          <button
            onClick={() => setIsEditing(!isEditing)}
            className="px-6 py-3 bg-white/20 hover:bg-white/30 backdrop-blur-sm rounded-xl font-semibold transition-all flex items-center space-x-2"
          >
            <Edit className="w-4 h-4" />
            <span>{isEditing ? "Done" : "Edit Picture"}</span>
          </button>
        </div>
      </div>

      {/* Contact Information - Always visible, read-only */}
      <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4">
          Contact Information
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-gray-400 text-sm mb-2">Email</label>
            <div className="flex items-center space-x-2 p-3 bg-slate-900/50 rounded-lg">
              <Mail className="w-5 h-5 text-gray-400" />
              <span className="text-white">{profile.email}</span>
            </div>
          </div>
          <div>
            <label className="block text-gray-400 text-sm mb-2">Phone</label>
            <div className="flex items-center space-x-2 p-3 bg-slate-900/50 rounded-lg">
              <Phone className="w-5 h-5 text-gray-400" />
              <span className="text-white">{profile.phone}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Performance Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center mb-4">
            <Phone className="w-5 h-5 text-blue-400" />
          </div>
          <p className="text-gray-400 text-sm mb-1">Total Calls</p>
          <p className="text-3xl font-bold text-white">{stats.totalCalls}</p>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center mb-4">
            <Star className="w-5 h-5 text-green-400" />
          </div>
          <p className="text-gray-400 text-sm mb-1">Avg Score</p>
          <p className="text-3xl font-bold text-white">{stats.avgScore}</p>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center mb-4">
            <Clock className="w-5 h-5 text-purple-400" />
          </div>
          <p className="text-gray-400 text-sm mb-1">Total Hours</p>
          <p className="text-3xl font-bold text-white">{stats.totalHours}h</p>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <div className="w-10 h-10 bg-orange-500/20 rounded-lg flex items-center justify-center mb-4">
            <TrendingUp className="w-5 h-5 text-orange-400" />
          </div>
          <p className="text-gray-400 text-sm mb-1">Current Streak</p>
          <p className="text-3xl font-bold text-white">{stats.currentStreak}</p>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Skill Radar */}
        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <h3 className="text-xl font-bold text-white mb-4 flex items-center">
            <Target className="w-5 h-5 mr-2 text-blue-400" />
            Skill Assessment
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={skillData}>
              <PolarGrid stroke="#334155" />
              <PolarAngleAxis dataKey="skill" stroke="#94a3b8" />
              <PolarRadiusAxis domain={[0, 100]} stroke="#94a3b8" />
              <Radar
                name="Score"
                dataKey="score"
                stroke="#3b82f6"
                fill="#3b82f6"
                fillOpacity={0.6}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Activity Chart */}
        <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6">
          <h3 className="text-xl font-bold text-white mb-4 flex items-center">
            <Activity className="w-5 h-5 mr-2 text-purple-400" />
            Monthly Activity
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={monthlyActivity}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="month" stroke="#94a3b8" />
              <YAxis yAxisId="left" stroke="#94a3b8" />
              <YAxis yAxisId="right" orientation="right" stroke="#94a3b8" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1e293b",
                  border: "1px solid #334155",
                  borderRadius: "8px",
                }}
              />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="calls"
                stroke="#3b82f6"
                strokeWidth={2}
                name="Calls"
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="score"
                stroke="#8b5cf6"
                strokeWidth={2}
                name="Score"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
