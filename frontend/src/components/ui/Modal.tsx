"use client";

import { useEffect, useRef } from "react";
import { X } from "lucide-react";

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
  title?: string;
  size?: "sm" | "md" | "lg" | "xl" | "2xl" | "3xl" | "4xl" | "5xl";
}

const sizeClasses = {
  sm: "max-w-sm",
  md: "max-w-md",
  lg: "max-w-lg",
  xl: "max-w-xl",
  "2xl": "max-w-2xl",
  "3xl": "max-w-3xl",
  "4xl": "max-w-4xl",
  "5xl": "max-w-5xl",
};

export default function Modal({
  isOpen,
  onClose,
  children,
  title,
  size = "2xl",
}: ModalProps) {
  const modalRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  // Escape key handler
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  // Focus trap: save previous focus, move focus into modal on open, restore on close
  useEffect(() => {
    if (isOpen) {
      previousFocusRef.current = document.activeElement as HTMLElement;
      modalRef.current?.focus();
    } else {
      previousFocusRef.current?.focus();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-black/20 backdrop-blur-[2px]"
        onClick={onClose}
      />
      <div
        ref={modalRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={title ? "modal-title" : undefined}
        tabIndex={-1}
        className={`relative bg-white rounded-xl p-6 w-full ${sizeClasses[size]} max-h-[90vh] overflow-y-auto shadow-xl focus:outline-none`}
        style={{ border: "1px solid var(--border)" }}
      >
        {title && (
          <div
            className="flex items-center justify-between mb-5 pb-4 border-b"
            style={{ borderColor: "var(--border)" }}
          >
            <h2
              id="modal-title"
              className="text-lg font-semibold"
              style={{ color: "var(--text-primary)" }}
            >
              {title}
            </h2>
            <button
              onClick={onClose}
              className="p-1.5 rounded-md hover:bg-[var(--hover-bg)] transition-colors"
              aria-label="Close dialog"
            >
              <X
                className="w-4 h-4"
                style={{ color: "var(--text-tertiary)" }}
              />
            </button>
          </div>
        )}
        {!title && (
          <button
            onClick={onClose}
            className="absolute top-4 right-4 p-1.5 rounded-md hover:bg-[var(--hover-bg)] transition-colors"
            aria-label="Close dialog"
          >
            <X className="w-4 h-4" style={{ color: "var(--text-tertiary)" }} />
          </button>
        )}
        {children}
      </div>
    </div>
  );
}
