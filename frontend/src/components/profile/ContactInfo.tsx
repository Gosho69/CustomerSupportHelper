"use client";

import { Mail, Phone } from "lucide-react";

interface ContactInfoProps {
  email: string;
  phone: string;
  isEditing: boolean;
  editedPhone: string;
  onPhoneChange: (phone: string) => void;
}

const cardStyle: React.CSSProperties = {
  background: "#ffffff",
  border: "1px solid var(--border, #e3e8ee)",
  borderRadius: "8px",
};

export default function ContactInfo({
  email,
  phone,
  isEditing,
  editedPhone,
  onPhoneChange,
}: ContactInfoProps) {
  return (
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
            <span style={{ color: "var(--text-primary)" }}>{email}</span>
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
              value={editedPhone}
              onChange={(e) => onPhoneChange(e.target.value)}
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
              <span style={{ color: "var(--text-primary)" }}>{phone}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
