"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Upload,
  Phone,
  CheckCircle,
  Clock,
  RefreshCw,
  ExternalLink,
  AlertCircle,
  PhoneCall,
} from "lucide-react";
import { mockCallCenterApi } from "@/lib/api";

interface ExternalCall {
  external_id: string;
  agent_email: string;
  caller_id: string;
  call_started_at: string;
  duration_seconds: number;
  analyzed: boolean;
  imported_at: string | null;
}

const ALLOWED_TYPES = new Set([
  "audio/wav",
  "audio/mpeg",
  "audio/mp3",
  "audio/x-m4a",
  "audio/m4a",
  "audio/flac",
  "audio/ogg",
  "audio/opus",
]);
const ALLOWED_EXTS = [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"];
const MAX_SIZE_MB = 100;

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function truncateId(id: string): string {
  return id.slice(0, 8) + "…";
}

export default function CallCenterPortal() {
  const [calls, setCalls] = useState<ExternalCall[]>([]);
  const [loadingCalls, setLoadingCalls] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Upload form state
  const [agentEmail, setAgentEmail] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const fetchCalls = useCallback(async (silent = false) => {
    if (!silent) setLoadingCalls(true);
    else setRefreshing(true);
    try {
      const res = await mockCallCenterApi.getCalls();
      setCalls(res.data || []);
    } catch {
      // keep stale data on error
    } finally {
      setLoadingCalls(false);
      setRefreshing(false);
    }
  }, []);

  // Initial load + auto-refresh every 10 s
  useEffect(() => {
    fetchCalls();
    const interval = setInterval(() => fetchCalls(true), 10_000);
    return () => clearInterval(interval);
  }, [fetchCalls]);

  // --- File handling ---
  function validateFile(f: File): string | null {
    const parts = f.name.split(".");
    const ext = parts.length === 1 ? "" : "." + parts.pop()!.toLowerCase();
    if (!ALLOWED_TYPES.has(f.type) && !(ext && ALLOWED_EXTS.includes(ext))) {
      return `Unsupported format. Allowed: ${ALLOWED_EXTS.join(", ")}`;
    }
    if (f.size > MAX_SIZE_MB * 1024 * 1024) {
      return `File too large (${(f.size / 1024 / 1024).toFixed(1)} MB). Max: ${MAX_SIZE_MB} MB`;
    }
    return null;
  }

  function handleFilePick(f: File) {
    const err = validateFile(f);
    if (err) {
      setUploadError(err);
      return;
    }
    setUploadError(null);
    setFile(f);
  }

  function onDragOver(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(true);
  }

  function onDragLeave() {
    setIsDragging(false);
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) handleFilePick(dropped);
  }

  function onFileInput(e: React.ChangeEvent<HTMLInputElement>) {
    const picked = e.target.files?.[0];
    if (picked) handleFilePick(picked);
    e.target.value = "";
  }

  // --- Upload ---
  async function handleUpload(e: React.FormEvent) {
    e.preventDefault();
    if (!file || !agentEmail) return;

    setUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    try {
      const formData = new FormData();
      formData.append("audio_file", file);
      formData.append("agent_email", agentEmail);
      const res = await mockCallCenterApi.uploadCall(formData);
      setUploadSuccess(
        `Call uploaded (ID: ${res.data.external_id.slice(0, 8)}…). It will be automatically imported and analyzed.`
      );
      setFile(null);
      setAgentEmail("");
      // Refresh the list to show the new entry
      await fetchCalls(true);
    } catch (err: any) {
      const detail =
        err?.response?.data?.audio_file?.[0] ||
        err?.response?.data?.agent_email?.[0] ||
        err?.response?.data?.detail ||
        "Upload failed. Check the API key and backend connection.";
      setUploadError(detail);
    } finally {
      setUploading(false);
    }
  }

  const pending = calls.filter((c) => !c.analyzed).length;
  const imported = calls.filter((c) => c.analyzed).length;

  return (
    <div className="min-h-screen" style={{ background: "var(--background)" }}>
      {/* Page header */}
      <div
        className="border-b px-8 py-5"
        style={{ background: "#ffffff", borderColor: "var(--border)" }}
      >
        <div className="flex items-center justify-between max-w-5xl">
          <div className="flex items-center gap-3">
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center"
              style={{ background: "#0e7490" }}
            >
              <PhoneCall className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1
                className="text-[18px] font-bold"
                style={{ color: "var(--text-primary)" }}
              >
                Call Center Simulator
              </h1>
              <p className="text-[13px]" style={{ color: "var(--text-secondary)" }}>
                External platform simulation — upload recordings here and AgentSights
                will auto-import and analyze them
              </p>
            </div>
          </div>
          {/* Stats strip */}
          <div className="flex items-center gap-6">
            <div className="text-center">
              <div
                className="text-[22px] font-bold"
                style={{ color: "var(--text-primary)" }}
              >
                {calls.length}
              </div>
              <div className="text-[11px]" style={{ color: "var(--text-secondary)" }}>
                Total
              </div>
            </div>
            <div className="text-center">
              <div className="text-[22px] font-bold" style={{ color: "#e68a00" }}>
                {pending}
              </div>
              <div className="text-[11px]" style={{ color: "var(--text-secondary)" }}>
                Pending
              </div>
            </div>
            <div className="text-center">
              <div className="text-[22px] font-bold" style={{ color: "#0caf60" }}>
                {imported}
              </div>
              <div className="text-[11px]" style={{ color: "var(--text-secondary)" }}>
                Imported
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="px-8 py-6 max-w-5xl space-y-6">
        {/* ── Upload card ── */}
        <div
          className="rounded-xl border overflow-hidden"
          style={{ background: "#ffffff", borderColor: "var(--border)" }}
        >
          <div
            className="px-6 py-4 border-b flex items-center gap-2"
            style={{ borderColor: "var(--border)" }}
          >
            <Upload className="w-4 h-4" style={{ color: "#0e7490" }} />
            <span
              className="text-[14px] font-semibold"
              style={{ color: "var(--text-primary)" }}
            >
              Upload Call Recording
            </span>
          </div>

          <form onSubmit={handleUpload} className="p-6 space-y-4">
            {/* Agent email */}
            <div>
              <label
                className="block text-[13px] font-medium mb-1.5"
                style={{ color: "var(--text-primary)" }}
              >
                Agent Email
              </label>
              <input
                type="email"
                required
                value={agentEmail}
                onChange={(e) => setAgentEmail(e.target.value)}
                placeholder="agent@yourcompany.com"
                className="w-full px-3 py-2 rounded-lg border text-[14px] outline-none transition-colors"
                style={{
                  borderColor: "var(--border)",
                  color: "var(--text-primary)",
                  background: "#fafbfc",
                }}
                onFocus={(e) =>
                  (e.currentTarget.style.borderColor = "#0e7490")
                }
                onBlur={(e) =>
                  (e.currentTarget.style.borderColor = "var(--border)")
                }
              />
            </div>

            {/* Drop zone */}
            <div>
              <label
                className="block text-[13px] font-medium mb-1.5"
                style={{ color: "var(--text-primary)" }}
              >
                Audio File
              </label>
              <div
                className={`border-2 border-dashed rounded-lg p-8 transition-all text-center cursor-pointer ${
                  isDragging ? "border-cyan-500 bg-cyan-50" : ""
                }`}
                style={{
                  borderColor: isDragging ? undefined : file ? "#0e7490" : "var(--border)",
                  background: isDragging ? undefined : file ? "#f0fdff" : "#fafbfc",
                }}
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onDrop={onDrop}
                onClick={() => inputRef.current?.click()}
              >
                <input
                  ref={inputRef}
                  type="file"
                  className="hidden"
                  accept={ALLOWED_EXTS.join(",")}
                  onChange={onFileInput}
                />
                {file ? (
                  <div className="flex flex-col items-center gap-1">
                    <Phone className="w-8 h-8" style={{ color: "#0e7490" }} />
                    <span
                      className="text-[14px] font-semibold mt-1"
                      style={{ color: "#0e7490" }}
                    >
                      {file.name}
                    </span>
                    <span
                      className="text-[12px]"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {(file.size / 1024 / 1024).toFixed(2)} MB — click to
                      change
                    </span>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-2">
                    <Upload className="w-8 h-8" style={{ color: "var(--text-tertiary)" }} />
                    <span
                      className="text-[14px] font-medium"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      Drop audio file here or{" "}
                      <span style={{ color: "#0e7490" }}>browse</span>
                    </span>
                    <span
                      className="text-[12px]"
                      style={{ color: "var(--text-tertiary)" }}
                    >
                      WAV, MP3, M4A, FLAC, OGG, OPUS · max 100 MB
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Feedback messages */}
            {uploadError && (
              <div
                className="flex items-center gap-2 px-3 py-2 rounded-lg text-[13px]"
                style={{ background: "#fff1f2", color: "#df1b41" }}
              >
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                {uploadError}
              </div>
            )}
            {uploadSuccess && (
              <div
                className="flex items-center gap-2 px-3 py-2 rounded-lg text-[13px]"
                style={{ background: "#f0fdf4", color: "#0caf60" }}
              >
                <CheckCircle className="w-4 h-4 flex-shrink-0" />
                {uploadSuccess}
              </div>
            )}

            <button
              type="submit"
              disabled={!file || !agentEmail || uploading}
              className="w-full py-2.5 rounded-lg text-[14px] font-semibold text-white transition-opacity disabled:opacity-50"
              style={{ background: "#0e7490" }}
            >
              {uploading ? "Uploading…" : "Upload to Call Center"}
            </button>
          </form>
        </div>

        {/* ── Calls table ── */}
        <div
          className="rounded-xl border overflow-hidden"
          style={{ background: "#ffffff", borderColor: "var(--border)" }}
        >
          <div
            className="px-6 py-4 border-b flex items-center justify-between"
            style={{ borderColor: "var(--border)" }}
          >
            <div className="flex items-center gap-2">
              <Phone className="w-4 h-4" style={{ color: "#0e7490" }} />
              <span
                className="text-[14px] font-semibold"
                style={{ color: "var(--text-primary)" }}
              >
                All Calls
              </span>
              <span
                className="text-[12px] px-2 py-0.5 rounded-full"
                style={{
                  background: "var(--accent-bg)",
                  color: "var(--accent)",
                }}
              >
                {calls.length}
              </span>
            </div>
            <button
              onClick={() => fetchCalls(true)}
              disabled={refreshing}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[13px] font-medium transition-colors hover:bg-[var(--hover-bg)]"
              style={{ color: "var(--text-secondary)" }}
            >
              <RefreshCw
                className={`w-3.5 h-3.5 ${refreshing ? "animate-spin" : ""}`}
              />
              Refresh
            </button>
          </div>

          {loadingCalls ? (
            <div
              className="flex items-center justify-center py-16 gap-2"
              style={{ color: "var(--text-secondary)" }}
            >
              <RefreshCw className="w-4 h-4 animate-spin" />
              <span className="text-[14px]">Loading calls…</span>
            </div>
          ) : calls.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 gap-3">
              <div
                className="w-12 h-12 rounded-full flex items-center justify-center"
                style={{ background: "var(--accent-bg)" }}
              >
                <PhoneCall className="w-6 h-6" style={{ color: "#0e7490" }} />
              </div>
              <p
                className="text-[14px] font-medium"
                style={{ color: "var(--text-secondary)" }}
              >
                No calls yet
              </p>
              <p
                className="text-[13px]"
                style={{ color: "var(--text-tertiary)" }}
              >
                Upload a recording above to get started
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr
                    className="border-b text-[12px] font-semibold uppercase tracking-wide"
                    style={{
                      borderColor: "var(--border)",
                      color: "var(--text-tertiary)",
                    }}
                  >
                    <th className="px-6 py-3 text-left">Call ID</th>
                    <th className="px-6 py-3 text-left">Agent</th>
                    <th className="px-6 py-3 text-left">Uploaded</th>
                    <th className="px-6 py-3 text-left">Duration</th>
                    <th className="px-6 py-3 text-left">Status</th>
                    <th className="px-6 py-3 text-left">Imported At</th>
                  </tr>
                </thead>
                <tbody>
                  {calls.map((call, i) => (
                    <tr
                      key={call.external_id}
                      className="border-b transition-colors hover:bg-[var(--hover-bg)]"
                      style={{
                        borderColor:
                          i === calls.length - 1
                            ? "transparent"
                            : "var(--border)",
                      }}
                    >
                      <td className="px-6 py-3">
                        <span
                          className="font-mono text-[12px] px-2 py-0.5 rounded"
                          style={{
                            background: "var(--accent-bg)",
                            color: "var(--accent)",
                          }}
                        >
                          {truncateId(call.external_id)}
                        </span>
                      </td>
                      <td
                        className="px-6 py-3 text-[13px]"
                        style={{ color: "var(--text-primary)" }}
                      >
                        {call.agent_email}
                      </td>
                      <td
                        className="px-6 py-3 text-[13px]"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        {formatDate(call.call_started_at)}
                      </td>
                      <td
                        className="px-6 py-3 text-[13px]"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        {call.duration_seconds > 0
                          ? formatDuration(call.duration_seconds)
                          : "—"}
                      </td>
                      <td className="px-6 py-3">
                        {call.analyzed ? (
                          <span
                            className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[12px] font-medium"
                            style={{
                              background: "#f0fdf4",
                              color: "#0caf60",
                            }}
                          >
                            <CheckCircle className="w-3 h-3" />
                            Imported
                          </span>
                        ) : (
                          <span
                            className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[12px] font-medium"
                            style={{
                              background: "#fffbeb",
                              color: "#e68a00",
                            }}
                          >
                            <Clock className="w-3 h-3" />
                            Pending
                          </span>
                        )}
                      </td>
                      <td
                        className="px-6 py-3 text-[13px]"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        {call.imported_at ? (
                          <div className="flex items-center gap-1.5">
                            {formatDate(call.imported_at)}
                            <a
                              href="/dashboard/calls"
                              className="inline-flex items-center gap-0.5 text-[12px] font-medium"
                              style={{ color: "#0e7490" }}
                              title="View analyzed call in AgentSights"
                            >
                              <ExternalLink className="w-3 h-3" />
                              View
                            </a>
                          </div>
                        ) : (
                          <span style={{ color: "var(--text-tertiary)" }}>
                            —
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Info banner */}
        <div
          className="rounded-xl border px-5 py-4 flex items-start gap-3"
          style={{ background: "#f0fdff", borderColor: "#a5f3fc" }}
        >
          <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" style={{ color: "#0e7490" }} />
          <div className="text-[13px]" style={{ color: "#164e63" }}>
            <span className="font-semibold">How it works:</span> Recordings
            uploaded here simulate an external call center system. AgentSights
            automatically polls for new calls every 5 minutes and runs the full
            analysis pipeline. Pending calls become available in{" "}
            <a
              href="/dashboard/calls"
              className="underline font-medium"
              style={{ color: "#0e7490" }}
            >
              My Calls / All Calls
            </a>{" "}
            once imported. The sync interval can be adjusted via{" "}
            <code
              className="px-1 py-0.5 rounded text-[12px]"
              style={{ background: "#cffafe" }}
            >
              CALL_SYNC_INTERVAL_SECONDS
            </code>{" "}
            in the backend environment.
          </div>
        </div>
      </div>
    </div>
  );
}
