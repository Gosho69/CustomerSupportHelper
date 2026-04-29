"use client";

import {
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Mic,
  ShieldCheck,
  Activity,
  Zap,
  Clock,
} from "lucide-react";
import {
  ComposedChart,
  Bar,
  Line,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { CallDetail, UtteranceStress, VocalAnalysis } from "./callDetailHelpers";

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function fmtTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

function mean(arr: number[]): number {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

function stdDev(arr: number[]): number {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
}

function normalizeArray(arr: number[]): number[] {
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  if (max === min) return arr.map(() => 0.5);
  return arr.map((v) => (v - min) / (max - min));
}

/**
 * WPM thresholds aligned with backend _assess_pacing():
 *   < 120        → too_slow
 *   120 – 180    → optimal
 *   181 – 220    → slightly_fast
 *   > 220        → too_fast
 */
function wpmCategory(wpm: number): { label: string; color: string; bg: string } {
  if (wpm < 120)  return { label: "Too slow",      color: "#f59e0b", bg: "#fffbeb" };
  if (wpm <= 180) return { label: "Optimal",       color: "#10b981", bg: "#d1fae5" };
  if (wpm <= 220) return { label: "Slightly fast", color: "#f59e0b", bg: "#fffbeb" };
  return            { label: "Too fast",          color: "#ef4444", bg: "#fef2f2" };
}

function scoreColor(v: number, thresholds = [75, 55]): string {
  if (v >= thresholds[0]) return "#10b981";
  if (v >= thresholds[1]) return "#f59e0b";
  return "#ef4444";
}

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

interface CriticalMoment {
  start: number;
  end: number;
  wpm: number;
  stressScore: number;
  reason: string;
}

interface PacingBucket {
  label: string;
  count: number;
  pct: number;
  color: string;
}

interface VoiceInsights {
  agentWpmAvg: number;
  agentRushedTurns: number;
  pacingBuckets: PacingBucket[];
  criticalMoments: CriticalMoment[];
  composureScore: number;
  calmResponseRate: number;
  highStressTurns: number;
  energyDeltaDb: number;
  coachingPriorities: string[];
}

// ─────────────────────────────────────────────────────────────────────────────
// Analytics — all thresholds match backend _assess_pacing()
// ─────────────────────────────────────────────────────────────────────────────

// Single "high stress" threshold used everywhere in this file.
const HIGH_STRESS = 0.60;

// Rush threshold aligned with backend "slightly_fast" boundary (>180 wpm).
const RUSHED_WPM = 180;

function deriveInsights(va: VocalAnalysis): VoiceInsights {
  const agents    = va.utterance_stress.filter(u => u.role === "Agent");
  const customers = va.utterance_stress.filter(u => u.role !== "Agent");
  const sorted    = [...va.utterance_stress].sort((a, b) => a.start - b.start);

  const agentWpms        = agents.filter(u => (u.speech_rate_wpm ?? 0) > 0).map(u => u.speech_rate_wpm!);
  const agentEnergies    = agents.filter(u => u.energy_db != null).map(u => u.energy_db!);
  const customerEnergies = customers.filter(u => u.energy_db != null).map(u => u.energy_db!);

  const agentWpmAvg   = Math.round(mean(agentWpms));
  const energyDeltaDb = parseFloat((mean(customerEnergies) - mean(agentEnergies)).toFixed(1));

  // ── Pacing distribution (buckets aligned with backend thresholds) ────────
  const agentTurnsWithWpm = agents.filter(u => (u.speech_rate_wpm ?? 0) > 0);
  const counts = { tooSlow: 0, optimal: 0, slightlyFast: 0, tooFast: 0 };
  for (const u of agentTurnsWithWpm) {
    const w = u.speech_rate_wpm!;
    if (w < 120)       counts.tooSlow++;
    else if (w <= 180) counts.optimal++;
    else if (w <= 220) counts.slightlyFast++;
    else               counts.tooFast++;
  }
  const total = agentTurnsWithWpm.length || 1;
  const pacingBuckets: PacingBucket[] = [
    { label: "Optimal (120–180 wpm)",   count: counts.optimal,       pct: Math.round(counts.optimal / total * 100),       color: "#10b981" },
    { label: "Slightly fast (181–220)", count: counts.slightlyFast,  pct: Math.round(counts.slightlyFast / total * 100),  color: "#f59e0b" },
    { label: "Too fast (>220 wpm)",     count: counts.tooFast,       pct: Math.round(counts.tooFast / total * 100),       color: "#ef4444" },
    { label: "Too slow (<120 wpm)",     count: counts.tooSlow,       pct: Math.round(counts.tooSlow / total * 100),       color: "#f59e0b" },
  ].filter(b => b.count > 0);

  const agentRushedTurns = agentTurnsWithWpm.filter(u => u.speech_rate_wpm! > RUSHED_WPM).length;

  // ── Critical convergence moments ─────────────────────────────────────────
  // Agent rushed (>180 wpm) right after a high-stress customer turn (>=0.60).
  const criticalMoments: CriticalMoment[] = [];
  for (let i = 1; i < sorted.length; i++) {
    const prev = sorted[i - 1];
    const cur  = sorted[i];
    if (
      cur.role === "Agent" &&
      (cur.speech_rate_wpm ?? 0) > RUSHED_WPM &&
      prev.role !== "Agent" &&
      prev.stress_score >= HIGH_STRESS
    ) {
      criticalMoments.push({
        start:       cur.start,
        end:         cur.end,
        wpm:         Math.round(cur.speech_rate_wpm!),
        stressScore: prev.stress_score,
        reason:      "Agent rushed after high-stress customer turn",
      });
    }
  }
  // Also flag agent turns that are both rushed and acoustically stressed themselves.
  for (const u of agents) {
    const alreadyListed = criticalMoments.some(m => Math.abs(m.start - u.start) < 0.5);
    if (!alreadyListed && (u.speech_rate_wpm ?? 0) > RUSHED_WPM && u.stress_score >= HIGH_STRESS) {
      criticalMoments.push({
        start:       u.start,
        end:         u.end,
        wpm:         Math.round(u.speech_rate_wpm!),
        stressScore: u.stress_score,
        reason:      "Agent rushed with elevated vocal stress",
      });
    }
  }
  criticalMoments.sort((a, b) => a.start - b.start);

  // ── Composure: low energy std-dev = steady voice ─────────────────────────
  const agentEnergyStd = stdDev(agentEnergies);
  // std ≤ 3 dB → 100, std ≥ 12 dB → 0 (linear)
  const composureScore = Math.round(Math.max(0, Math.min(100, ((12 - agentEnergyStd) / 9) * 100)));

  // ── De-escalation rate ────────────────────────────────────────────────────
  // When customer stress >= HIGH_STRESS, does agent respond at lower energy?
  let highStressTurns = 0;
  let calmResponses   = 0;
  for (let i = 0; i < sorted.length - 1; i++) {
    const cur  = sorted[i];
    const next = sorted[i + 1];
    if (cur.role !== "Agent" && cur.stress_score >= HIGH_STRESS && next.role === "Agent") {
      highStressTurns++;
      if ((next.energy_db ?? -30) < (cur.energy_db ?? -30)) calmResponses++;
    }
  }
  const calmResponseRate = highStressTurns > 0
    ? Math.round((calmResponses / highStressTurns) * 100)
    : 100;

  // ── Coaching priorities ───────────────────────────────────────────────────
  const coaching: string[] = [];

  if (criticalMoments.length > 0) {
    const first = criticalMoments[0];
    coaching.push(
      `At ${fmtTime(first.start)}, speech rate hit ${first.wpm} wpm right after the customer's ` +
      `stress peaked (${Math.round(first.stressScore * 100)}%). ` +
      `Slowing below 160 wpm in these moments signals control, not urgency.`
    );
  }

  if (agentRushedTurns > 2) {
    coaching.push(
      `${agentRushedTurns} turns exceeded ${RUSHED_WPM} wpm (audio-measured speech rate). ` +
      `Check the Transcript tab at those timestamps — rapid speech under pressure often sounds dismissive.`
    );
  }

  if (calmResponseRate < 60 && highStressTurns > 1) {
    coaching.push(
      `Only ${calmResponseRate}% of high-stress customer turns were met with a quieter agent voice. ` +
      `Lowering voice energy — not just changing words — is the fastest acoustic de-escalation signal.`
    );
  }

  if (energyDeltaDb < 0) {
    coaching.push(
      `Agent voice energy matched or exceeded the customer's on average. ` +
      `A quieter, steadier tone than the customer's creates a de-escalating contrast.`
    );
  }

  if (coaching.length === 0 && agentWpmAvg > 0) {
    const cat = wpmCategory(agentWpmAvg);
    coaching.push(
      cat.label === "Optimal"
        ? `Audio-measured speech rate averaged ${agentWpmAvg} wpm — within the 120–180 wpm optimal range.`
        : `Audio-measured speech rate averaged ${agentWpmAvg} wpm (${cat.label.toLowerCase()}). ` +
          `The Behavior tab shows the text-based WPM which may differ slightly.`
    );
  }

  return {
    agentWpmAvg, agentRushedTurns, pacingBuckets, criticalMoments,
    composureScore, calmResponseRate, highStressTurns, energyDeltaDb,
    coachingPriorities: coaching,
  };
}

function buildChartData(utterances: UtteranceStress[]) {
  const energyVals = utterances.filter(u => u.energy_db != null).map(u => u.energy_db!);
  const norm = normalizeArray(energyVals);
  let nIdx = 0;
  return utterances.map(u => ({
    time:       u.start,
    timeEnd:    u.end,
    role:       u.role,
    energy:     u.energy_db,
    energyNorm: u.energy_db != null ? norm[nIdx++] : null,
    wpm:        u.speech_rate_wpm,
    stress:     u.stress_score,
  }));
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────────────────

function KpiCard({
  icon, label, value, unit = "", sub, color, bg,
}: {
  icon: React.ReactNode; label: string; value: string | number;
  unit?: string; sub?: string; color: string; bg?: string;
}) {
  return (
    <div className="rounded-lg p-4 flex flex-col gap-1"
      style={{ background: bg ?? "var(--surface)", border: "1px solid var(--border)" }}>
      <div className="flex items-center gap-2 mb-1">
        <span style={{ color }}>{icon}</span>
        <span className="text-xs font-medium uppercase tracking-wide"
          style={{ color: "var(--text-tertiary)" }}>{label}</span>
      </div>
      <span className="text-2xl font-bold" style={{ color }}>
        {value}
        <span className="text-base font-normal ml-1"
          style={{ color: "var(--text-secondary)" }}>{unit}</span>
      </span>
      {sub && <span className="text-xs" style={{ color: "var(--text-tertiary)" }}>{sub}</span>}
    </div>
  );
}

function ChartTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;
  return (
    <div className="rounded-lg p-3 text-xs shadow-lg"
      style={{ background: "var(--surface)", border: "1px solid var(--border)",
               color: "var(--text-primary)", minWidth: 170 }}>
      <div className="font-semibold mb-1.5">{d.role} · {fmtTime(d.time)} – {fmtTime(d.timeEnd)}</div>
      {d.energy != null && (
        <div className="flex justify-between gap-3">
          <span style={{ color: "var(--text-secondary)" }}>Voice energy</span>
          <span className="font-medium">{d.energy.toFixed(1)} dB</span>
        </div>
      )}
      {d.wpm != null && d.wpm > 0 && (
        <div className="flex justify-between gap-3 mt-0.5">
          <span style={{ color: "var(--text-secondary)" }}>Speech rate (audio)</span>
          <span className="font-medium"
            style={{ color: d.wpm > 220 ? "#ef4444" : d.wpm > 180 ? "#f59e0b" : "#10b981" }}>
            {Math.round(d.wpm)} wpm
          </span>
        </div>
      )}
      <div className="flex justify-between gap-3 mt-0.5">
        <span style={{ color: "var(--text-secondary)" }}>Acoustic stress</span>
        <span className="font-medium">{Math.round(d.stress * 100)}%</span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────────

export default function VoiceAnalysisTab({ call }: { call: CallDetail }) {
  const va = call.vocal_analysis;

  if (!va) {
    return (
      <div className="flex items-center justify-center py-16 text-center">
        <div>
          <AlertTriangle className="w-10 h-10 mx-auto mb-3"
            style={{ color: "var(--text-tertiary)" }} />
          <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
            Voice analysis data is not available for this call.
          </p>
        </div>
      </div>
    );
  }

  if (va.analysis_error) {
    return (
      <div className="flex items-start gap-3 rounded-lg p-4"
        style={{ background: "#fef3c7", border: "1px solid #f59e0b" }}>
        <AlertTriangle className="w-5 h-5 mt-0.5 flex-shrink-0" style={{ color: "#92400e" }} />
        <div>
          <p className="font-semibold text-sm" style={{ color: "#92400e" }}>
            Voice analysis unavailable
          </p>
          <p className="text-sm mt-0.5" style={{ color: "#78350f" }}>{va.analysis_error}</p>
        </div>
      </div>
    );
  }

  const ins      = deriveInsights(va);
  const allBars  = buildChartData(va.utterance_stress);
  const peakTime = va.peak_stress?.start ?? null;

  const composureColor   = scoreColor(ins.composureScore);
  const calmColor        = scoreColor(ins.calmResponseRate);
  const agentStressPct   = Math.round(va.agent_stress_avg * 100);
  const custStressPct    = Math.round(va.customer_stress_avg * 100);
  const agentStressColor = scoreColor(100 - agentStressPct, [60, 40]);   // lower stress = better
  const custStressColor  = custStressPct >= 65 ? "#ef4444" : custStressPct >= 45 ? "#f59e0b" : "#10b981";

  return (
    <div className="space-y-5">

      {/* ── What this tab measures ────────────────────────────────────────── */}
      <div className="flex items-start gap-3 rounded-lg px-4 py-3"
        style={{ background: "var(--accent-bg)", border: "1px solid var(--border)" }}>
        <Activity className="w-4 h-4 mt-0.5 flex-shrink-0" style={{ color: "var(--accent)" }} />
        <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
          <strong style={{ color: "var(--text-primary)" }}>Acoustic analysis</strong>
          {" "}— what the audio signal reveals beyond the transcript.
          The emotion model reads words; this reads the waveform — energy, frequency, and speech rate
          per utterance. A customer saying <em>"that's fine"</em> while raising their voice gets
          caught here, not in the text. Speech rate here is measured from audio segment duration
          (see the <strong>Behavior tab</strong> for the text-based aggregate WPM).
        </p>
      </div>

      {/* ── Acoustic scorecard — 4 KPIs, all from vocal_analysis ─────────── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard
          icon={<Activity className="w-4 h-4" />}
          label="Agent Stress"
          value={agentStressPct}
          unit="%"
          sub="Relative to this call's baseline"
          color={agentStressColor}
        />
        <KpiCard
          icon={<Activity className="w-4 h-4" />}
          label="Customer Stress"
          value={custStressPct}
          unit="%"
          sub={custStressPct >= 65 ? "Elevated" : custStressPct >= 45 ? "Moderate" : "Low"}
          color={custStressColor}
        />
        <KpiCard
          icon={<ShieldCheck className="w-4 h-4" />}
          label="Vocal Composure"
          value={ins.composureScore}
          unit="%"
          sub="Agent energy consistency"
          color={composureColor}
        />
        <KpiCard
          icon={<Mic className="w-4 h-4" />}
          label="De-escalation"
          value={ins.calmResponseRate}
          unit="%"
          sub={`${ins.highStressTurns} high-stress turns`}
          color={calmColor}
        />
      </div>

      {/* ── Pacing Intelligence ───────────────────────────────────────────── */}
      <div className="rounded-lg p-5 space-y-4"
        style={{ border: "1px solid var(--border)", background: "var(--surface)" }}>
        <div className="flex items-center gap-2">
          <Zap className="w-4 h-4" style={{ color: "var(--accent)" }} />
          <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
            Pacing Intelligence
          </h3>
        </div>
        <p className="text-xs -mt-1" style={{ color: "var(--text-tertiary)" }}>
          Speech rate measured from audio segment duration.
          For text-based aggregate WPM, see the <strong>Behavior tab</strong>.
          Thresholds match the backend analyzer: optimal = 120–180 wpm.
        </p>

        {/* Pacing distribution */}
        {ins.pacingBuckets.length > 0 && (
          <div>
            <p className="text-xs font-medium mb-2" style={{ color: "var(--text-secondary)" }}>
              Distribution across agent turns
            </p>
            <div className="space-y-1.5">
              {ins.pacingBuckets.map((b) => (
                <div key={b.label} className="flex items-center gap-3">
                  <span className="text-xs w-36 flex-shrink-0"
                    style={{ color: "var(--text-tertiary)" }}>{b.label}</span>
                  <div className="flex-1 rounded-full overflow-hidden h-2.5"
                    style={{ background: "var(--border)" }}>
                    <div className="h-full rounded-full"
                      style={{ width: `${b.pct}%`, background: b.color }} />
                  </div>
                  <span className="text-xs w-10 text-right flex-shrink-0 font-medium"
                    style={{ color: b.color }}>{b.pct}%</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Critical convergence moments */}
        {ins.criticalMoments.length > 0 ? (
          <div>
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="w-4 h-4 flex-shrink-0" style={{ color: "#ef4444" }} />
              <p className="text-xs font-semibold" style={{ color: "#991b1b" }}>
                {ins.criticalMoments.length} moment{ins.criticalMoments.length > 1 ? "s" : ""} where
                agent speech rate was elevated while customer stress was high
              </p>
            </div>
            <div className="space-y-1.5">
              {ins.criticalMoments.slice(0, 5).map((m, i) => (
                <div key={i} className="flex items-center gap-3 rounded-md px-3 py-2 text-xs"
                  style={{ background: "#fef2f2", border: "1px solid #fecaca" }}>
                  <span className="font-mono font-semibold flex-shrink-0"
                    style={{ color: "#991b1b" }}>{fmtTime(m.start)}</span>
                  <span className="px-1.5 py-0.5 rounded text-xs font-medium flex-shrink-0"
                    style={{ background: "#fee2e2", color: "#7f1d1d" }}>{m.wpm} wpm</span>
                  <span style={{ color: "#991b1b", flex: 1 }}>{m.reason}</span>
                  <span className="flex-shrink-0 font-medium" style={{ color: "#ef4444" }}>
                    cust. stress {Math.round(m.stressScore * 100)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-2 rounded-md px-3 py-2 text-xs"
            style={{ background: "#d1fae5" }}>
            <span style={{ color: "#065f46" }}>✓</span>
            <span style={{ color: "#065f46" }}>
              No critical pacing moments — agent speech rate was not elevated during customer stress peaks.
            </span>
          </div>
        )}
      </div>

      {/* ── Acoustic coaching priorities ─────────────────────────────────── */}
      {ins.coachingPriorities.length > 0 && (
        <div className="rounded-lg p-5"
          style={{ border: "1px solid var(--border)", background: "var(--surface)" }}>
          <div className="flex items-center gap-2 mb-3">
            <Clock className="w-4 h-4" style={{ color: "var(--accent)" }} />
            <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
              Acoustic Coaching Observations
            </h3>
            <span className="text-xs px-2 py-0.5 rounded-full font-medium ml-auto"
              style={{ background: "#dbeafe", color: "#1d4ed8" }}>
              Supplements the Coaching tab
            </span>
          </div>
          <ol className="space-y-2.5">
            {ins.coachingPriorities.map((tip, i) => (
              <li key={i} className="flex gap-3 rounded-md px-3 py-2.5 text-sm"
                style={{
                  background: i === 0 ? "#fff7ed" : "var(--accent-bg)",
                  border: `1px solid ${i === 0 ? "#fed7aa" : "var(--border)"}`,
                }}>
                <span className="flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center
                  text-xs font-bold mt-0.5"
                  style={{ background: i === 0 ? "#f97316" : "var(--accent)", color: "#fff" }}>
                  {i + 1}
                </span>
                <span style={{ color: i === 0 ? "#7c2d12" : "var(--text-primary)" }}>{tip}</span>
              </li>
            ))}
          </ol>
        </div>
      )}

      {/* ── Voice energy & speech rate timeline ──────────────────────────── */}
      <div className="rounded-lg p-5"
        style={{ border: "1px solid var(--border)", background: "var(--surface)" }}>
        <div className="flex items-start justify-between mb-1">
          <div>
            <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
              Voice Energy &amp; Speech Rate Timeline
            </h3>
            <p className="text-xs mt-0.5" style={{ color: "var(--text-tertiary)" }}>
              Bars = normalised voice energy (blue = agent, red = customer, bright = high stress).
              Line = audio-measured speech rate. Green band = 120–180 wpm optimal.
              Red dots = exceeded {RUSHED_WPM} wpm.
            </p>
          </div>
          <div className="flex items-center gap-3 text-xs flex-shrink-0"
            style={{ color: "var(--text-secondary)" }}>
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-3 rounded-sm bg-blue-400 opacity-70" />Agent
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-3 rounded-sm bg-red-400 opacity-70" />Customer
            </span>
          </div>
        </div>

        <ResponsiveContainer width="100%" height={290}>
          <ComposedChart data={allBars}
            margin={{ top: 16, right: 44, left: 0, bottom: 4 }} barCategoryGap={0}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
            <XAxis dataKey="time" type="number" domain={["dataMin", "dataMax"]}
              tickFormatter={fmtTime}
              tick={{ fontSize: 11, fill: "var(--text-tertiary)" }} tickCount={9} />
            <YAxis yAxisId="energy" domain={[0, 1]}
              tickFormatter={(v) => `${Math.round(v * 100)}%`}
              tick={{ fontSize: 11, fill: "var(--text-tertiary)" }} width={44}
              label={{ value: "Energy", angle: -90, position: "insideLeft",
                       offset: 12, fontSize: 10, fill: "var(--text-tertiary)" }} />
            <YAxis yAxisId="wpm" orientation="right" domain={[0, 280]}
              tick={{ fontSize: 11, fill: "var(--text-tertiary)" }} width={36}
              label={{ value: "wpm", angle: 90, position: "insideRight",
                       offset: -4, fontSize: 10, fill: "var(--text-tertiary)" }} />
            <Tooltip content={<ChartTooltip />} />

            {/* Optimal WPM band — 120 to 180, matching backend thresholds */}
            <ReferenceLine yAxisId="wpm" y={180} stroke="#10b981" strokeDasharray="4 2"
              strokeOpacity={0.6}
              label={{ value: "180", position: "right", fontSize: 9, fill: "#10b981" }} />
            <ReferenceLine yAxisId="wpm" y={120} stroke="#10b981" strokeDasharray="4 2"
              strokeOpacity={0.6}
              label={{ value: "120", position: "right", fontSize: 9, fill: "#10b981" }} />

            {peakTime !== null && (
              <ReferenceLine yAxisId="energy" x={peakTime}
                stroke="#f59e0b" strokeWidth={2} strokeDasharray="4 2"
                label={{ value: "Peak stress", position: "top",
                         fontSize: 10, fill: "#92400e" }} />
            )}

            <Bar yAxisId="energy" dataKey="energyNorm" maxBarSize={18}
              radius={[3, 3, 0, 0]} name="Voice energy">
              {allBars.map((e, i) => (
                <Cell key={i}
                  fill={e.stress >= 0.70 ? "#ef4444" : e.role === "Agent" ? "#60a5fa" : "#f87171"}
                  fillOpacity={e.stress >= 0.70 ? 0.9 : 0.55} />
              ))}
            </Bar>

            <Line yAxisId="wpm" type="monotone" dataKey="wpm"
              stroke="#a78bfa" strokeWidth={1.5}
              dot={(props: any) => {
                const { cx, cy, payload } = props;
                if (!payload.wpm || payload.wpm <= 0) return <g key={props.key} />;
                const rushed = payload.wpm > RUSHED_WPM;
                return <circle key={props.key} cx={cx} cy={cy}
                  r={rushed ? 4 : 2}
                  fill={rushed ? "#ef4444" : "#a78bfa"} stroke="none" />;
              }}
              name="Speech rate (audio)" connectNulls={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* ── Top stress moments ────────────────────────────────────────────── */}
      <div className="rounded-lg p-5"
        style={{ border: "1px solid var(--border)", background: "var(--surface)" }}>
        <h3 className="text-sm font-semibold mb-1" style={{ color: "var(--text-primary)" }}>
          Highest Acoustic Stress Moments
        </h3>
        <p className="text-xs mb-3" style={{ color: "var(--text-tertiary)" }}>
          Cross-reference with the Transcript and Emotions tabs to validate.
          Stress scores are call-relative (50% = average for this call).
        </p>
        <div className="space-y-2">
          {[...va.utterance_stress]
            .sort((a, b) => b.stress_score - a.stress_score)
            .slice(0, 4)
            .map((u, i) => (
              <div key={i} className="flex items-center gap-3 rounded-lg px-4 py-3 text-sm"
                style={{
                  background: u.stress_score >= 0.70 ? "#fef2f2"
                            : u.stress_score >= 0.58 ? "#fffbeb"
                            : "var(--accent-bg)",
                  border: `1px solid ${
                    u.stress_score >= 0.70 ? "#fecaca"
                    : u.stress_score >= 0.58 ? "#fde68a"
                    : "var(--border)"}`,
                }}>
                <span className="font-black text-base w-5 text-center flex-shrink-0"
                  style={{ color: u.stress_score >= 0.70 ? "#ef4444" : "#f59e0b" }}>
                  {i + 1}
                </span>
                <span className="text-xs font-semibold px-2 py-0.5 rounded flex-shrink-0"
                  style={{
                    background: u.role === "Agent" ? "#dbeafe" : "#fee2e2",
                    color:      u.role === "Agent" ? "#1d4ed8"  : "#991b1b",
                  }}>
                  {u.role}
                </span>
                <span className="font-mono text-xs flex-shrink-0"
                  style={{ color: "var(--text-secondary)" }}>
                  {fmtTime(u.start)} – {fmtTime(u.end)}
                </span>
                <div className="flex gap-3 ml-auto flex-shrink-0 text-xs"
                  style={{ color: "var(--text-tertiary)" }}>
                  {u.speech_rate_wpm != null && u.speech_rate_wpm > 0 && (
                    <span style={{ color: u.speech_rate_wpm > RUSHED_WPM ? "#ef4444" : "inherit" }}>
                      {Math.round(u.speech_rate_wpm)} wpm
                    </span>
                  )}
                  {u.energy_db != null && <span>{u.energy_db.toFixed(1)} dB</span>}
                  <span className="font-semibold"
                    style={{ color: u.stress_score >= 0.70 ? "#ef4444" : "#f59e0b" }}>
                    {Math.round(u.stress_score * 100)}%
                  </span>
                </div>
              </div>
            ))}
        </div>
      </div>

      {/* ── Trend banners ─────────────────────────────────────────────────── */}
      {va.stress_escalation_detected && (
        <div className="flex items-start gap-3 rounded-lg p-4"
          style={{ background: "#fef3c7", border: "1px solid #f59e0b" }}>
          <TrendingUp className="w-5 h-5 mt-0.5 flex-shrink-0" style={{ color: "#92400e" }} />
          <p className="text-sm font-medium" style={{ color: "#92400e" }}>
            Customer vocal energy increased toward the end of the call.
            Review the final 30% of the Transcript tab for missed de-escalation opportunities.
          </p>
        </div>
      )}
      {va.calm_recovery_detected && (
        <div className="flex items-start gap-3 rounded-lg p-4"
          style={{ background: "#d1fae5", border: "1px solid #10b981" }}>
          <TrendingDown className="w-5 h-5 mt-0.5 flex-shrink-0" style={{ color: "#065f46" }} />
          <p className="text-sm font-medium" style={{ color: "#065f46" }}>
            Customer vocal stress recovered after the peak — acoustic evidence of successful de-escalation.
          </p>
        </div>
      )}
    </div>
  );
}
