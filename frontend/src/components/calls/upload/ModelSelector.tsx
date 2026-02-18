import React from "react";

interface ModelSelectorProps {
  selectedModel: "gpt4" | "local";
  onSelectModel: (model: "gpt4" | "local") => void;
}

export default function ModelSelector({
  selectedModel,
  onSelectModel,
}: ModelSelectorProps) {
  return (
    <div
      className="mb-4 p-4 rounded-lg"
      style={{
        background: "var(--background)",
        border: "1px solid var(--border)",
      }}
    >
      <p
        className="text-sm font-medium mb-3"
        style={{ color: "var(--text-primary)" }}
      >
        Summarization Model
      </p>
      <div className="flex space-x-3">
        <button
          onClick={() => onSelectModel("gpt4")}
          className={`flex-1 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
            selectedModel === "gpt4" ? "ring-2 shadow-sm" : "hover:bg-gray-50"
          }`}
          style={{
            background:
              selectedModel === "gpt4" ? "var(--accent-bg)" : "#ffffff",
            border: `1px solid ${
              selectedModel === "gpt4" ? "var(--accent)" : "var(--border)"
            }`,
            color:
              selectedModel === "gpt4"
                ? "var(--accent)"
                : "var(--text-secondary)",
            ...(selectedModel === "gpt4" ? { ringColor: "var(--accent)" } : {}),
          }}
        >
          <div className="flex items-center justify-center space-x-2">
            <span>OpenAI (GPT-4)</span>
          </div>
          <p className="text-xs mt-1" style={{ color: "var(--text-tertiary)" }}>
            Better quality, requires API key
          </p>
        </button>
        <button
          onClick={() => onSelectModel("local")}
          className={`flex-1 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
            selectedModel === "local" ? "ring-2 shadow-sm" : "hover:bg-gray-50"
          }`}
          style={{
            background:
              selectedModel === "local" ? "var(--accent-bg)" : "#ffffff",
            border: `1px solid ${
              selectedModel === "local" ? "var(--accent)" : "var(--border)"
            }`,
            color:
              selectedModel === "local"
                ? "var(--accent)"
                : "var(--text-secondary)",
          }}
        >
          <div className="flex items-center justify-center space-x-2">
            <span>Local AI</span>
          </div>
          <p className="text-xs mt-1" style={{ color: "var(--text-tertiary)" }}>
            Runs locally, no API key needed
          </p>
        </button>
      </div>
    </div>
  );
}
