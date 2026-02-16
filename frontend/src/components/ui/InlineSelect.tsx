"use client";

import { useState, useRef, useEffect } from "react";
import { ChevronDown, Check } from "lucide-react";

interface InlineSelectOption {
  value: string;
  label: string;
}

interface InlineSelectProps {
  options: InlineSelectOption[];
  value: string;
  onChange: (value: string) => void;
  className?: string;
}

export default function InlineSelect({
  options,
  value,
  onChange,
  className = "",
}: InlineSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const selectedOption = options.find((opt) => opt.value === value);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSelect = (optionValue: string) => {
    onChange(optionValue);
    setIsOpen(false);
  };

  return (
    <div ref={ref} className={`relative ${className}`}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:ring-offset-1"
        style={{
          background: "#ffffff",
          border: "1px solid var(--border)",
          color: "var(--text-primary)",
        }}
      >
        <span>{selectedOption?.label || "Select"}</span>
        <ChevronDown
          className={`w-4 h-4 transition-transform duration-200 ${isOpen ? "rotate-180" : ""}`}
          style={{ color: "var(--text-tertiary)" }}
        />
      </button>

      {isOpen && (
        <div
          className="absolute right-0 mt-1 min-w-[180px] bg-white rounded-lg shadow-lg border overflow-hidden z-[10000]"
          style={{ borderColor: "var(--border)" }}
        >
          <div className="py-1">
            {options.map((option) => (
              <button
                key={option.value}
                type="button"
                onClick={() => handleSelect(option.value)}
                className="w-full px-3 py-2 text-left text-sm flex items-center justify-between hover:bg-[var(--hover-bg)] transition-colors"
                style={{
                  color:
                    value === option.value
                      ? "var(--accent)"
                      : "var(--text-primary)",
                  background:
                    value === option.value ? "var(--accent-bg)" : undefined,
                }}
              >
                <span className="font-medium">{option.label}</span>
                {value === option.value && (
                  <Check
                    className="w-4 h-4"
                    style={{ color: "var(--accent)" }}
                  />
                )}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
