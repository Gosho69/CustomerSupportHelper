import { Upload } from "lucide-react";
import { useRouter } from "next/navigation";

export default function CallsHeader() {
  const router = useRouter();

  return (
    <div className="flex items-center justify-between">
      <div>
        <h1 className="text-3xl font-bold text-white">My Calls</h1>
        <p className="text-gray-400 mt-1">
          View and analyze all your call recordings
        </p>
      </div>
      <button
        onClick={() => router.push("/dashboard/upload-call")}
        className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-semibold rounded-xl transition-all shadow-lg shadow-purple-500/30 flex items-center space-x-2"
      >
        <Upload className="w-5 h-5" />
        <span>Upload Call</span>
      </button>
    </div>
  );
}
