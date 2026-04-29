"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { X, Phone, Clock, Calendar } from "lucide-react";
import { callsApi } from "@/lib/api";
import { CallDetail, formatDuration, formatDate } from "./callDetailHelpers";
import OverviewTab from "./OverviewTab";
import TranscriptTab from "./TranscriptTab";
import EmotionsTab from "./EmotionsTab";
import BehaviorTab from "./BehaviorTab";
import CoachingTab from "./CoachingTab";
import VoiceAnalysisTab from "./VoiceAnalysisTab";

interface CallDetailModalProps {
  callId: number;
  onClose: () => void;
}

type TabId = "overview" | "transcript" | "emotions" | "behavior" | "coaching" | "voice";

const TABS: { id: TabId; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "transcript", label: "Transcript" },
  { id: "emotions", label: "Emotions" },
  { id: "behavior", label: "Behavior" },
  { id: "coaching", label: "Coaching" },
  { id: "voice", label: "Voice Analysis" },
];

export default function CallDetailModal({
  callId,
  onClose,
}: CallDetailModalProps) {
  const [call, setCall] = useState<CallDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<TabId>("overview");
  const [portalRoot, setPortalRoot] = useState<HTMLElement | null>(null);
  const modalRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  // Escape key handler
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  // Focus trap: save previous focus, restore on unmount
  useEffect(() => {
    previousFocusRef.current = document.activeElement as HTMLElement;
    modalRef.current?.focus();
    return () => {
      previousFocusRef.current?.focus();
    };
  }, []);

  useEffect(() => {
    let root = document.getElementById("modal-portal");
    if (!root) {
      root = document.createElement("div");
      root.id = "modal-portal";
      document.body.appendChild(root);
    }
    setPortalRoot(root);

    return () => {
      if (root && root.childNodes.length === 0) {
        document.body.removeChild(root);
      }
    };
  }, []);

  useEffect(() => {
    fetchCallDetail();
  }, [callId]);

  const fetchCallDetail = async () => {
    try {
      setLoading(true);
      const response = await callsApi.getCallDetail(callId);
      setCall(response.data);
    } catch (error) {
      console.error("Failed to fetch call details:", error);
    } finally {
      setLoading(false);
    }
  };

  const renderTabContent = () => {
    if (!call) return null;

    switch (activeTab) {
      case "overview":
        return <OverviewTab call={call} />;
      case "transcript":
        return <TranscriptTab call={call} />;
      case "emotions":
        return <EmotionsTab call={call} />;
      case "behavior":
        return <BehaviorTab call={call} />;
      case "coaching":
        return <CoachingTab call={call} />;
      case "voice":
        return <VoiceAnalysisTab call={call} />;
    }
  };

  if (!portalRoot) return null;

  return createPortal(
    <div
      className="fixed inset-0 bg-black/20 z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        ref={modalRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="call-detail-title"
        tabIndex={-1}
        className="bg-white rounded-lg w-full max-w-6xl max-h-[90vh] overflow-hidden flex flex-col shadow-xl focus:outline-none"
        style={{ border: "1px solid var(--border)" }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          className="p-6 flex items-center justify-between"
          style={{ borderBottom: "1px solid var(--border)" }}
        >
          <div className="flex items-center space-x-4">
            <div
              className="w-12 h-12 rounded-lg flex items-center justify-center"
              style={{ background: "var(--accent-bg)" }}
            >
              <Phone className="w-6 h-6" style={{ color: "var(--accent)" }} />
            </div>
            <div>
              <h2
                id="call-detail-title"
                className="text-2xl font-bold"
                style={{ color: "var(--text-primary)" }}
              >
                Call #{callId} Details
              </h2>
              {call && (
                <div
                  className="flex items-center space-x-4 mt-1 text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
                  <span className="flex items-center">
                    <Calendar className="w-4 h-4 mr-1" />
                    {formatDate(call.call_date)}
                  </span>
                  <span className="flex items-center">
                    <Clock className="w-4 h-4 mr-1" />
                    {formatDuration(call.duration)}
                  </span>
                </div>
              )}
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X className="w-6 h-6" style={{ color: "var(--text-tertiary)" }} />
          </button>
        </div>

        {/* Tabs */}
        <div
          className="px-6 pt-4"
          style={{ borderBottom: "1px solid var(--border)" }}
        >
          <div className="flex space-x-1">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className="px-6 py-3 rounded-t-lg font-medium transition-all"
                style={{
                  color:
                    activeTab === tab.id
                      ? "var(--accent)"
                      : "var(--text-secondary)",
                  borderBottom:
                    activeTab === tab.id
                      ? "2px solid var(--accent)"
                      : "2px solid transparent",
                  background:
                    activeTab === tab.id ? "var(--accent-bg)" : "transparent",
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
            </div>
          ) : (
            renderTabContent()
          )}
        </div>
      </div>
    </div>,
    portalRoot,
  );
}
