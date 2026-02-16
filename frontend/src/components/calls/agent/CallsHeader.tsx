import { Upload } from "lucide-react";
import { useRouter } from "next/navigation";

export default function CallsHeader() {
  const router = useRouter();

  return (
    <div className="flex items-center justify-between">
      <div>
        <h1
          className="text-3xl font-bold"
          style={{ color: "var(--text-primary)" }}
        >
          My Calls
        </h1>
        <p className="mt-1" style={{ color: "var(--text-secondary)" }}>
          View and analyze all your call recordings
        </p>
      </div>
      <button
        onClick={() => router.push("/dashboard/upload-call")}
        className="px-6 py-3 font-semibold rounded-lg transition-all flex items-center space-x-2"
        style={{ background: "var(--accent)", color: "#ffffff" }}
      >
        <Upload className="w-5 h-5" />
        <span>Upload Call</span>
      </button>
    </div>
  );
}
