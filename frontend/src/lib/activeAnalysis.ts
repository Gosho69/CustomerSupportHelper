export interface StoredCall {
  callId: number;
  fileName: string;
  fileSize: number;
  currentStepIndex: number;
}

interface ActiveAnalysisData {
  inProgress: boolean;
  calls: StoredCall[];
}

const STORAGE_KEY = "csh_active_analysis";

export function getActiveAnalysis(): ActiveAnalysisData | null {
  if (typeof window === "undefined") return null;
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return null;
    return JSON.parse(stored) as ActiveAnalysisData;
  } catch {
    return null;
  }
}

function save(data: ActiveAnalysisData): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
}

export function markAnalysisStarted(): void {
  const existing = getActiveAnalysis();
  if (existing?.inProgress) return;
  save({ inProgress: true, calls: [] });
}

export function addStoredCall(
  callId: number,
  fileName: string,
  fileSize: number,
): void {
  const existing = getActiveAnalysis() ?? { inProgress: true, calls: [] };
  const calls = existing.calls.filter((c) => c.fileName !== fileName);
  calls.push({ callId, fileName, fileSize, currentStepIndex: 0 });
  save({ ...existing, calls });
}

export function updateStoredCallStep(
  fileName: string,
  stepIndex: number,
): void {
  const data = getActiveAnalysis();
  if (!data) return;
  const calls = data.calls.map((c) =>
    c.fileName === fileName ? { ...c, currentStepIndex: stepIndex } : c,
  );
  save({ ...data, calls });
}

export function getStoredCallId(fileName: string): number | null {
  const data = getActiveAnalysis();
  if (!data) return null;
  const call = data.calls.find((c) => c.fileName === fileName);
  return call ? call.callId : null;
}

export function getStoredStep(fileName: string): number | null {
  const data = getActiveAnalysis();
  if (!data) return null;
  const call = data.calls.find((c) => c.fileName === fileName);
  return call?.currentStepIndex ?? null;
}

export function isAnalysisInProgress(): boolean {
  const data = getActiveAnalysis();
  return data?.inProgress === true;
}

export function clearActiveAnalysis(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(STORAGE_KEY);
}
