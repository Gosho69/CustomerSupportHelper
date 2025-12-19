import { FileText } from "lucide-react";

export default function EmptyState() {
  return (
    <div className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-12 text-center">
      <FileText className="w-16 h-16 text-gray-600 mx-auto mb-4" />
      <h3 className="text-xl font-bold text-white mb-2">No reports found</h3>
      <p className="text-gray-400">
        Try adjusting your filters or search query
      </p>
    </div>
  );
}
