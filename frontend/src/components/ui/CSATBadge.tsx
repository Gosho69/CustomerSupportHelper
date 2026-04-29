interface CSATBadgeProps {
  score: number | null | undefined;
  label: string | null | undefined;
}

const LABEL_DISPLAY: Record<string, string> = {
  very_satisfied:    "Very satisfied",
  satisfied:         "Satisfied",
  neutral:           "Neutral",
  dissatisfied:      "Dissatisfied",
  very_dissatisfied: "Very dissatisfied",
};

export default function CSATBadge({ score, label }: CSATBadgeProps) {
  if (score == null || label == null) return null;

  const style =
    score >= 4.0
      ? { background: "var(--success-bg)", color: "var(--success)" }
      : score >= 3.0
      ? { background: "var(--warning-bg)", color: "var(--warning)" }
      : { background: "var(--danger-bg)",  color: "var(--danger)"  };

  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium"
      style={style}
      title={LABEL_DISPLAY[label] ?? label}
    >
      <span>CSAT</span>
      <span className="font-bold">{score.toFixed(1)}</span>
    </span>
  );
}
