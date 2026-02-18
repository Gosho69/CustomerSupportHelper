"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { createPortal } from "react-dom";
import { Check, ChevronDown, Search } from "lucide-react";

interface Option {
  value: string;
  label: string;
  subtitle?: string;
}

interface CustomSelectProps {
  options: Option[];
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  label?: string;
  required?: boolean;
  searchable?: boolean;
}

export default function CustomSelect({
  options,
  value,
  onChange,
  placeholder = "Select an option",
  label,
  required = false,
  searchable = true,
}: CustomSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [dropdownPos, setDropdownPos] = useState<{
    top: number;
    left: number;
    width: number;
  } | null>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  const selectedOption = options.find((opt) => opt.value === value);

  const filteredOptions = searchable
    ? options.filter(
        (option) =>
          option.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
          option.subtitle?.toLowerCase().includes(searchQuery.toLowerCase()),
      )
    : options;

  // Calculate dropdown position from trigger button
  const updatePosition = useCallback(() => {
    if (triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect();
      setDropdownPos({
        top: rect.bottom + 4,
        left: rect.left,
        width: rect.width,
      });
    }
  }, []);

  // Close on outside click
  useEffect(() => {
    if (!isOpen) return;

    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node;
      if (
        triggerRef.current?.contains(target) ||
        dropdownRef.current?.contains(target)
      ) {
        return;
      }
      setIsOpen(false);
      setSearchQuery("");
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isOpen]);

  // Recalculate position on scroll / resize while open
  useEffect(() => {
    if (!isOpen) return;

    updatePosition();

    const handleUpdate = () => updatePosition();
    window.addEventListener("scroll", handleUpdate, true);
    window.addEventListener("resize", handleUpdate);
    return () => {
      window.removeEventListener("scroll", handleUpdate, true);
      window.removeEventListener("resize", handleUpdate);
    };
  }, [isOpen, updatePosition]);

  // Auto-focus search when dropdown opens
  useEffect(() => {
    if (isOpen && searchable) {
      // Small delay so portal has rendered
      requestAnimationFrame(() => searchInputRef.current?.focus());
    }
  }, [isOpen, searchable]);

  const handleToggle = () => {
    if (!isOpen) {
      updatePosition();
    }
    setIsOpen((prev) => !prev);
    if (isOpen) setSearchQuery("");
  };

  const handleSelect = (optionValue: string) => {
    onChange(optionValue);
    setIsOpen(false);
    setSearchQuery("");
  };

  const dropdownContent = isOpen && dropdownPos && (
    <div
      ref={dropdownRef}
      className="bg-white border rounded-md shadow-lg max-h-[300px] overflow-hidden"
      style={{
        position: "fixed",
        top: dropdownPos.top,
        left: dropdownPos.left,
        width: dropdownPos.width,
        zIndex: 200000,
        borderColor: "var(--border)",
      }}
    >
      {searchable && (
        <div className="p-2 border-b" style={{ borderColor: "var(--border)" }}>
          <div className="relative">
            <Search
              className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4"
              style={{ color: "var(--text-tertiary)" }}
            />
            <input
              ref={searchInputRef}
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search..."
              className="w-full pl-9 pr-4 py-1.5 bg-white border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
              style={{
                borderColor: "var(--border)",
                color: "var(--text-primary)",
              }}
              onMouseDown={(e) => e.stopPropagation()}
            />
          </div>
        </div>
      )}

      <div className="overflow-y-auto max-h-[240px] py-1">
        {filteredOptions.length === 0 ? (
          <div
            className="px-4 py-3 text-sm text-center"
            style={{ color: "var(--text-tertiary)" }}
          >
            No options found
          </div>
        ) : (
          filteredOptions.map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() => handleSelect(option.value)}
              className={`w-full px-4 py-2.5 text-left hover:bg-[var(--hover-bg)] transition-colors flex items-center justify-between group ${
                value === option.value ? "bg-[var(--accent-bg)]" : ""
              }`}
            >
              <div className="flex-1">
                <div
                  className="font-medium text-sm"
                  style={{
                    color:
                      value === option.value
                        ? "var(--accent)"
                        : "var(--text-primary)",
                  }}
                >
                  {option.label}
                </div>
                {option.subtitle && (
                  <div
                    className="text-xs mt-0.5"
                    style={{ color: "var(--text-tertiary)" }}
                  >
                    {option.subtitle}
                  </div>
                )}
              </div>
              {value === option.value && (
                <Check
                  className="w-4 h-4 ml-2"
                  style={{ color: "var(--accent)" }}
                />
              )}
            </button>
          ))
        )}
      </div>
    </div>
  );

  return (
    <div className="relative">
      {label && (
        <label
          className="block text-sm font-medium mb-2"
          style={{ color: "var(--text-secondary)" }}
        >
          {label}
          {required && (
            <span
              className="ml-0.5"
              style={{ color: "var(--danger, #e53935)" }}
            >
              *
            </span>
          )}
        </label>
      )}

      <button
        ref={triggerRef}
        type="button"
        onClick={handleToggle}
        className="w-full px-4 bg-white border rounded-md text-left flex items-center justify-between hover:bg-[var(--hover-bg)] transition-all focus:outline-none focus:ring-2 focus:ring-[var(--accent)] h-[72px]"
        style={{ borderColor: "var(--border)" }}
      >
        <div className="flex-1 flex flex-col justify-center min-h-[48px]">
          {selectedOption ? (
            <div>
              <div
                className="font-medium truncate leading-tight"
                style={{ color: "var(--text-primary)" }}
              >
                {selectedOption.label}
              </div>
              {selectedOption.subtitle && (
                <div
                  className="text-sm mt-0.5 truncate"
                  style={{ color: "var(--text-tertiary)" }}
                >
                  {selectedOption.subtitle}
                </div>
              )}
            </div>
          ) : (
            <span style={{ color: "var(--text-tertiary)" }}>{placeholder}</span>
          )}
        </div>
        <ChevronDown
          className={`w-5 h-5 transition-transform ${
            isOpen ? "rotate-180" : ""
          }`}
        />
      </button>

      {typeof document !== "undefined" &&
        createPortal(dropdownContent, document.body)}
    </div>
  );
}
