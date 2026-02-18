import React from "react";
import { CheckCircle, AlertCircle } from "lucide-react";

export default function UploadTips() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div
        className="rounded-lg p-6"
        style={{ background: "#ffffff", border: "1px solid var(--border)" }}
      >
        <div className="flex items-start space-x-3">
          <div className="w-10 h-10 bg-blue-50 rounded-lg flex items-center justify-center flex-shrink-0">
            <CheckCircle
              className="w-5 h-5"
              style={{ color: "var(--accent)" }}
            />
          </div>
          <div>
            <h4
              className="font-semibold mb-2"
              style={{ color: "var(--text-primary)" }}
            >
              Best Practices
            </h4>
            <ul
              className="text-sm space-y-2"
              style={{ color: "var(--text-secondary)" }}
            >
              <li>• Ensure clear audio quality for accurate analysis</li>
              <li>• Upload calls within 24 hours for timely feedback</li>
              <li>• Include complete conversations (intro to outro)</li>
              <li>• Avoid background noise when possible</li>
            </ul>
          </div>
        </div>
      </div>

      <div
        className="rounded-lg p-6"
        style={{ background: "#ffffff", border: "1px solid var(--border)" }}
      >
        <div className="flex items-start space-x-3">
          <div
            className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0"
            style={{ background: "var(--accent-bg)" }}
          >
            <AlertCircle
              className="w-5 h-5"
              style={{ color: "var(--accent)" }}
            />
          </div>
          <div>
            <h4
              className="font-semibold mb-2"
              style={{ color: "var(--text-primary)" }}
            >
              What You&apos;ll Get
            </h4>
            <ul
              className="text-sm space-y-2"
              style={{ color: "var(--text-secondary)" }}
            >
              <li>• Emotional sentiment analysis</li>
              <li>• Behavioral pattern insights</li>
              <li>• Personalized coaching tips</li>
              <li>• Performance scoring</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
