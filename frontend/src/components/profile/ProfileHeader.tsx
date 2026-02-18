"use client";

import { Building2, Calendar, Edit } from "lucide-react";

interface ProfileHeaderProps {
  firstName: string;
  lastName: string;
  role: string;
  company: string;
  joinDate: string;
  isEditing: boolean;
  saving: boolean;
  editedProfile: {
    first_name: string;
    last_name: string;
    email: string;
    phone: string;
  };
  setEditedProfile: (profile: any) => void;
  onSave: () => void;
  onCancel: () => void;
  onStartEdit: () => void;
}

export default function ProfileHeader({
  firstName,
  lastName,
  role,
  company,
  joinDate,
  isEditing,
  saving,
  editedProfile,
  setEditedProfile,
  onSave,
  onCancel,
  onStartEdit,
}: ProfileHeaderProps) {
  return (
    <div
      className="rounded-lg p-8 text-white"
      style={{ background: "var(--accent)" }}
    >
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between">
        <div className="flex items-center space-x-6 mb-6 md:mb-0">
          <div className="relative">
            <div className="w-24 h-24 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center text-4xl font-bold border-4 border-white/30">
              {firstName[0]}
              {lastName[0]}
            </div>
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
                {firstName} {lastName}
              </h1>
            )}
            <p className="text-white/80 mb-1">{role}</p>
            <div className="flex items-center space-x-4 text-sm text-white/70">
              <span className="flex items-center">
                <Building2 className="w-4 h-4 mr-1" />
                {company}
              </span>
              <span className="flex items-center">
                <Calendar className="w-4 h-4 mr-1" />
                Joined {joinDate}
              </span>
            </div>
          </div>
        </div>
        <div className="flex gap-2">
          {isEditing ? (
            <>
              <button
                onClick={onSave}
                disabled={saving}
                className="px-6 py-3 bg-white rounded-lg font-semibold transition-all flex items-center space-x-2 disabled:opacity-60 disabled:cursor-not-allowed"
                style={{ color: "var(--accent)" }}
              >
                {saving ? (
                  <>
                    <svg
                      className="animate-spin h-4 w-4"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                    <span>Saving...</span>
                  </>
                ) : (
                  <span>Save</span>
                )}
              </button>
              <button
                onClick={onCancel}
                disabled={saving}
                className="px-6 py-3 bg-white/20 hover:bg-white/30 backdrop-blur-sm rounded-lg font-semibold transition-all flex items-center space-x-2 disabled:opacity-60 disabled:cursor-not-allowed"
              >
                <span>Cancel</span>
              </button>
            </>
          ) : (
            <button
              onClick={onStartEdit}
              className="px-6 py-3 bg-white/20 hover:bg-white/30 rounded-lg font-semibold transition-all flex items-center space-x-2"
            >
              <Edit className="w-4 h-4" />
              <span>Edit Profile</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
