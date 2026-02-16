import { FileText } from "lucide-react";

export default function EmptyState() {
  return (
    <div
      className="rounded-lg p-12 text-center"
      style={{ background: "#ffffff", border: "1px solid var(--border)" }}
    >
      <FileText
        className="w-16 h-16 mx-auto mb-4"
        style={{ color: "var(--text-secondary)" }}
      />
      <h3
        className="text-xl font-bold mb-2"
        style={{ color: "var(--text-primary)" }}
      >
        No reports found
      </h3>
      <p style={{ color: "var(--text-secondary)" }}>
        Try adjusting your filters or search query
      </p>
    </div>
  );
}
