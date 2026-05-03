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
  BarChart2,
  X,
} from "lucide-react";
import { callsApi, mockCallCenterApi } from "@/lib/api";
import { useAuthStore } from "@/store/authStore";
import type {
  BulkUploadCallResult,
  BulkUploadResponse,
  QueueStatus,
} from "@/lib/api";

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

function formatBytes(bytes: number): string {
  return (bytes / 1024 / 1024).toFixed(2) + " MB";
}

export default function CallCenterPortal() {
  const { user } = useAuthStore();
  const currentEmail = user?.email ?? "";

  const [calls, setCalls] = useState<ExternalCall[]>([]);
  const [loadingCalls, setLoadingCalls] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Upload form state — pre-filled with the logged-in user's email
  const [agentEmail, setAgentEmail] = useState(currentEmail);
  const [files, setFiles] = useState<File[]>([]);
  const [fileResults, setFileResults] = useState<
    Map<string, BulkUploadCallResult>
  >(new Map());
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Queue status state
  const [queueStatus, setQueueStatus] = useState<QueueStatus | null>(null);

  const fetchCalls = useCallback(async (silent = false) => {
    if (!silent) setLoadingCalls(true);
    else setRefreshing(true);
    try {
      const res = await mockCallCenterApi.getCalls(undefined, currentEmail);
      setCalls(res.data || []);
    } catch {
      // keep stale data on error
    } finally {
      setLoadingCalls(false);
      setRefreshing(false);
    }
  }, [currentEmail]);

  const fetchQueueStatus = useCallback(async () => {
    try {
      const res = await callsApi.getQueueStatus();
      setQueueStatus(res.data);
    } catch {
      // non-critical — keep stale data
    }
  }, []);

  // Initial load + auto-refresh every 10 s
  useEffect(() => {
    fetchCalls();
    fetchQueueStatus();
    const interval = setInterval(() => {
      fetchCalls(true);
      fetchQueueStatus();
    }, 10_000);
    return () => clearInterval(interval);
  }, [fetchCalls, fetchQueueStatus]);

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

  function handleFilePick(newFiles: FileList | File[]) {
    const valid: File[] = [];
    const errors: string[] = [];
    Array.from(newFiles).forEach((f) => {
      const err = validateFile(f);
      if (err) errors.push(`${f.name}: ${err}`);
      else valid.push(f);
    });
    if (errors.length) setUploadError(errors.join("\n"));
    else setUploadError(null);
    if (valid.length) setFiles((prev) => [...prev, ...valid]);
  }

  function removeFile(target: File) {
    setFiles((prev) => prev.filter((f) => f !== target));
    setFileResults((prev) => {
      const next = new Map(prev);
      next.delete(target.name);
      return next;
    });
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
    if (e.dataTransfer.files.length) handleFilePick(e.dataTransfer.files);
  }

  function onFileInput(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files?.length) handleFilePick(e.target.files);
    e.target.value = "";
  }

  // --- Upload ---
  async function handleUpload(e: React.FormEvent) {
    e.preventDefault();
    if (files.length === 0 || !agentEmail) return;

    setUploading(true);
    setUploadError(null);
    setUploadSuccess(null);
    setFileResults(new Map());

    try {
      const formData = new FormData();
      formData.append("agent_email", agentEmail);
      files.forEach((f) => formData.append("audio_files", f));

      const res = await mockCallCenterApi.bulkUploadCalls(formData);
      const data: BulkUploadResponse = res.data;

      const resultMap = new Map<string, BulkUploadCallResult>();
      data.calls.forEach((r) => resultMap.set(r.filename, r));
      setFileResults(resultMap);

      if (data.failed === 0) {
        setUploadSuccess(
          `${data.imported} call${data.imported > 1 ? "s" : ""} queued for analysis. Processing will begin within seconds.`,
        );
        setFiles([]);
        setAgentEmail(currentEmail);
      } else if (data.imported > 0) {
        setUploadSuccess(
          `${data.imported} of ${data.total} calls queued. ${data.failed} failed — see details below.`,
        );
      } else {
        setUploadError(
          `All ${data.total} files failed to upload. See details below.`,
        );
      }

      await Promise.all([fetchCalls(true), fetchQueueStatus()]);
    } catch (err: unknown) {
      const e = err as {
        response?: { data?: Record<string, string[]> };
      };
      const detail =
        e?.response?.data?.agent_email?.[0] ||
        e?.response?.data?.audio_files?.[0] ||
        (e?.response?.data?.detail as string | undefined) ||
        "Upload failed. Check the API key and backend connection.";
      setUploadError(detail);
    } finally {
      setUploading(false);
    }
  }

  const pending = calls.filter((c) => !c.analyzed).length;
  const imported = calls.filter((c) => c.analyzed).length;

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
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
            <p
              className="text-[13px]"
              style={{ color: "var(--text-secondary)" }}
            >
              External platform simulation — upload recordings here and
              AgentSights will auto-import and analyze them
            </p>
          </div>
        </div>
        {/* Stats strip */}
        <div className="flex items-center gap-6 flex-shrink-0">
          <div className="text-center">
            <div
              className="text-[22px] font-bold"
              style={{ color: "var(--text-primary)" }}
            >
              {calls.length}
            </div>
            <div
              className="text-[11px]"
              style={{ color: "var(--text-secondary)" }}
            >
              Total
            </div>
          </div>
          <div className="text-center">
            <div
              className="text-[22px] font-bold"
              style={{ color: "#e68a00" }}
            >
              {pending}
            </div>
            <div
              className="text-[11px]"
              style={{ color: "var(--text-secondary)" }}
            >
              Pending
            </div>
          </div>
          <div className="text-center">
            <div
              className="text-[22px] font-bold"
              style={{ color: "#0caf60" }}
            >
              {imported}
            </div>
            <div
              className="text-[11px]"
              style={{ color: "var(--text-secondary)" }}
            >
              Imported
            </div>
          </div>
        </div>
      </div>

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
              Upload Call Recordings
            </span>
          </div>

          <form onSubmit={handleUpload} className="p-6 space-y-4">
            {/* Agent email — auto-filled from logged-in account */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label
                  className="block text-[13px] font-medium"
                  style={{ color: "var(--text-primary)" }}
                >
                  Agent Email
                </label>
                <span className="text-[11px]" style={{ color: "var(--text-tertiary)" }}>
                  auto-filled from your account
                </span>
              </div>
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
                onFocus={(e) => (e.currentTarget.style.borderColor = "#0e7490")}
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
                Audio Files
              </label>

              {/* Drop target */}
              <div
                className={`border-2 border-dashed rounded-lg p-6 transition-all text-center cursor-pointer ${
                  isDragging ? "border-cyan-500 bg-cyan-50" : ""
                }`}
                style={{
                  borderColor: isDragging
                    ? undefined
                    : files.length > 0
                      ? "#0e7490"
                      : "var(--border)",
                  background: isDragging
                    ? undefined
                    : files.length > 0
                      ? "#f0fdff"
                      : "#fafbfc",
                }}
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onDrop={onDrop}
                onClick={() => inputRef.current?.click()}
              >
                <input
                  ref={inputRef}
                  type="file"
                  multiple
                  className="hidden"
                  accept={ALLOWED_EXTS.join(",")}
                  onChange={onFileInput}
                />
                <div className="flex flex-col items-center gap-2">
                  <Upload
                    className="w-6 h-6"
                    style={{
                      color: files.length > 0 ? "#0e7490" : "var(--text-tertiary)",
                    }}
                  />
                  <span
                    className="text-[13px] font-medium"
                    style={{
                      color: files.length > 0 ? "#0e7490" : "var(--text-secondary)",
                    }}
                  >
                    {files.length > 0
                      ? `${files.length} file${files.length > 1 ? "s" : ""} selected — drop more or click to add`
                      : <>Drop audio files here or <span style={{ color: "#0e7490" }}>browse</span></>}
                  </span>
                  <span
                    className="text-[12px]"
                    style={{ color: "var(--text-tertiary)" }}
                  >
                    WAV, MP3, M4A, FLAC, OGG, OPUS · max 100 MB each · up to 50
                    files
                  </span>
                </div>
              </div>

              {/* File list */}
              {files.length > 0 && (
                <div className="mt-3 space-y-1.5">
                  {files.map((f) => {
                    const result = fileResults.get(f.name);
                    return (
                      <div
                        key={f.name + f.size}
                        className="flex items-center justify-between px-3 py-2 rounded-lg border text-[13px]"
                        style={{
                          borderColor: result?.error
                            ? "#fca5a5"
                            : result?.external_id
                              ? "#86efac"
                              : "var(--border)",
                          background: result?.error
                            ? "#fff1f2"
                            : result?.external_id
                              ? "#f0fdf4"
                              : "#fafbfc",
                        }}
                      >
                        <div className="flex items-center gap-2 min-w-0">
                          <Phone
                            className="w-3.5 h-3.5 flex-shrink-0"
                            style={{
                              color: result?.error
                                ? "#df1b41"
                                : result?.external_id
                                  ? "#0caf60"
                                  : "#0e7490",
                            }}
                          />
                          <span
                            className="truncate font-medium"
                            style={{ color: "var(--text-primary)" }}
                          >
                            {f.name}
                          </span>
                          <span
                            className="flex-shrink-0 text-[12px]"
                            style={{ color: "var(--text-tertiary)" }}
                          >
                            {formatBytes(f.size)}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 flex-shrink-0 ml-3">
                          {result?.external_id && (
                            <span
                              className="text-[11px] font-medium px-2 py-0.5 rounded-full"
                              style={{ background: "#dcfce7", color: "#0caf60" }}
                            >
                              Queued · {truncateId(result.external_id)}
                            </span>
                          )}
                          {result?.error && (
                            <span
                              className="text-[11px] font-medium px-2 py-0.5 rounded-full"
                              style={{ background: "#fff1f2", color: "#df1b41" }}
                            >
                              {result.error}
                            </span>
                          )}
                          {!result && (
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                removeFile(f);
                              }}
                              className="p-0.5 rounded hover:bg-[var(--hover-bg)] transition-colors"
                              style={{ color: "var(--text-tertiary)" }}
                              title="Remove file"
                            >
                              <X className="w-3.5 h-3.5" />
                            </button>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Feedback messages */}
            {uploadError && (
              <div
                className="flex items-start gap-2 px-3 py-2 rounded-lg text-[13px] whitespace-pre-line"
                style={{ background: "#fff1f2", color: "#df1b41" }}
              >
                <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
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
              disabled={files.length === 0 || !agentEmail || uploading}
              className="w-full py-2.5 rounded-lg text-[14px] font-semibold text-white transition-opacity disabled:opacity-50"
              style={{ background: "#0e7490" }}
            >
              {uploading
                ? `Uploading ${files.length} file${files.length > 1 ? "s" : ""}…`
                : `Upload ${files.length > 0 ? files.length + " " : ""}Call${files.length !== 1 ? "s" : ""} to Call Center`}
            </button>
          </form>
        </div>

        {/* ── Processing queue status ── */}
        {queueStatus !== null && (
          <div
            className="rounded-xl border overflow-hidden"
            style={{ background: "#ffffff", borderColor: "var(--border)" }}
          >
            <div
              className="px-6 py-4 border-b flex items-center gap-2"
              style={{ borderColor: "var(--border)" }}
            >
              <BarChart2 className="w-4 h-4" style={{ color: "#0e7490" }} />
              <span
                className="text-[14px] font-semibold"
                style={{ color: "var(--text-primary)" }}
              >
                Processing Queue
              </span>
              <span
                className="text-[12px] ml-auto"
                style={{ color: "var(--text-tertiary)" }}
              >
                updates every 10 s
              </span>
            </div>
            <div className="px-6 py-4 grid grid-cols-4 gap-4">
              {/* Awaiting import */}
              <div
                className="rounded-lg px-4 py-3 flex flex-col gap-1"
                style={{ background: "#fffbeb" }}
              >
                <span className="text-[11px] font-semibold uppercase tracking-wide" style={{ color: "#e68a00" }}>
                  Awaiting Import
                </span>
                <span className="text-[28px] font-bold leading-none" style={{ color: "#e68a00" }}>
                  {queueStatus.awaiting_import}
                </span>
                <span className="text-[11px]" style={{ color: "#92400e" }}>
                  {queueStatus.in_queue > 0
                    ? `+ ${queueStatus.in_queue} in analysis queue`
                    : "in call center, not imported"}
                </span>
              </div>
              {/* Processing */}
              <div
                className="rounded-lg px-4 py-3 flex flex-col gap-1"
                style={{ background: "#eff6ff" }}
              >
                <span className="text-[11px] font-semibold uppercase tracking-wide" style={{ color: "#2563eb" }}>
                  Processing
                </span>
                <div className="flex items-center gap-2">
                  <span className="text-[28px] font-bold leading-none" style={{ color: "#2563eb" }}>
                    {queueStatus.processing}
                  </span>
                  {queueStatus.processing > 0 && (
                    <RefreshCw className="w-4 h-4 animate-spin" style={{ color: "#2563eb" }} />
                  )}
                </div>
                <span className="text-[11px]" style={{ color: "#1e40af" }}>
                  being analyzed now
                </span>
              </div>
              {/* Completed today */}
              <div
                className="rounded-lg px-4 py-3 flex flex-col gap-1"
                style={{ background: "#f0fdf4" }}
              >
                <span className="text-[11px] font-semibold uppercase tracking-wide" style={{ color: "#0caf60" }}>
                  Completed Today
                </span>
                <span className="text-[28px] font-bold leading-none" style={{ color: "#0caf60" }}>
                  {queueStatus.completed_today}
                </span>
                <span className="text-[11px]" style={{ color: "#166534" }}>
                  analysis done
                </span>
              </div>
              {/* Failed today */}
              <div
                className="rounded-lg px-4 py-3 flex flex-col gap-1"
                style={{ background: queueStatus.failed_today > 0 ? "#fff1f2" : "#fafbfc" }}
              >
                <span
                  className="text-[11px] font-semibold uppercase tracking-wide"
                  style={{ color: queueStatus.failed_today > 0 ? "#df1b41" : "var(--text-tertiary)" }}
                >
                  Failed Today
                </span>
                <span
                  className="text-[28px] font-bold leading-none"
                  style={{ color: queueStatus.failed_today > 0 ? "#df1b41" : "var(--text-tertiary)" }}
                >
                  {queueStatus.failed_today}
                </span>
                <span
                  className="text-[11px]"
                  style={{ color: queueStatus.failed_today > 0 ? "#9f1239" : "var(--text-tertiary)" }}
                >
                  {queueStatus.failed_today > 0 ? "check call logs" : "no failures"}
                </span>
              </div>
            </div>
          </div>
        )}

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
                Upload recordings above to get started
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
          <AlertCircle
            className="w-4 h-4 mt-0.5 flex-shrink-0"
            style={{ color: "#0e7490" }}
          />
          <div className="text-[13px]" style={{ color: "#164e63" }}>
            <span className="font-semibold">How it works:</span> Upload one or
            many recordings at once — each batch is queued immediately and
            AgentSights starts processing within seconds. Calls are analyzed
            one by one to keep resource usage predictable. Analyzed calls appear
            in{" "}
            <a
              href="/dashboard/calls"
              className="underline font-medium"
              style={{ color: "#0e7490" }}
            >
              My Calls / All Calls
            </a>{" "}
            once complete. The polling interval can be adjusted via{" "}
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
  );
}


