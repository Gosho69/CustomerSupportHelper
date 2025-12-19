import { TrendingUp, TrendingDown } from "lucide-react";

interface PerformanceHeaderProps {
  latestScore: number;
  avgScore: number;
  totalReports: number;
  scoreTrend: number;
}

export default function PerformanceHeader({
  latestScore,
  avgScore,
  totalReports,
  scoreTrend,
}: PerformanceHeaderProps) {
  return (
    <div className="bg-gradient-to-r from-purple-600 to-pink-600 rounded-2xl p-8 text-white">
      <h1 className="text-3xl font-bold mb-2">My Performance Reports</h1>
      <p className="text-purple-100 mb-6">
        Track your progress and identify areas for improvement
      </p>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
          <p className="text-purple-100 text-sm mb-1">Current Score</p>
          <p className="text-3xl font-bold">{latestScore}</p>
        </div>
        <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
          <p className="text-purple-100 text-sm mb-1">Average Score</p>
          <p className="text-3xl font-bold">{Math.round(avgScore)}</p>
        </div>
        <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
          <p className="text-purple-100 text-sm mb-1">Total Reports</p>
          <p className="text-3xl font-bold">{totalReports}</p>
        </div>
        <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
          <p className="text-purple-100 text-sm mb-1">Trend</p>
          <div className="flex items-center space-x-2">
            <p className="text-3xl font-bold">
              {scoreTrend > 0 ? "+" : ""}
              {scoreTrend}
            </p>
            {scoreTrend > 0 ? (
              <TrendingUp className="w-6 h-6 text-green-300" />
            ) : (
              <TrendingDown className="w-6 h-6 text-red-300" />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
