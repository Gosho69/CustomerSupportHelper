"use client";

import { useEffect } from "react";
import { createPortal } from "react-dom";
import { AlertTriangle, Trash2, X } from "lucide-react";

interface ConfirmDialogProps {
  open: boolean;
  title?: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: "danger" | "warning" | "default";
  onConfirm: () => void;
  onCancel: () => void;
}

export default function ConfirmDialog({
  open,
  title = "Are you sure?",
  message,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  variant = "danger",
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  useEffect(() => {
    if (open) {
      const handleEsc = (e: KeyboardEvent) => {
        if (e.key === "Escape") onCancel();
      };
      document.addEventListener("keydown", handleEsc);
      return () => document.removeEventListener("keydown", handleEsc);
    }
  }, [open, onCancel]);

  if (!open || typeof document === "undefined") return null;

  const variantConfig = {
    danger: {
      iconBg: "var(--danger-bg)",
      iconColor: "var(--danger)",
      btnBg: "var(--danger)",
      btnHoverBg: "#c4162e",
      btnText: "#ffffff",
      Icon: Trash2,
    },
    warning: {
      iconBg: "var(--warning-bg)",
      iconColor: "var(--warning)",
      btnBg: "var(--warning)",
      btnHoverBg: "#cc7a00",
      btnText: "#ffffff",
      Icon: AlertTriangle,
    },
    default: {
      iconBg: "var(--accent-bg)",
      iconColor: "var(--accent)",
      btnBg: "var(--accent)",
      btnHoverBg: "var(--accent-light)",
      btnText: "#ffffff",
      Icon: AlertTriangle,
    },
  };

  const config = variantConfig[variant];
  const IconComponent = config.Icon;

  return createPortal(
    <div className="fixed inset-0 z-[999999] flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-black/30 backdrop-blur-[2px]"
        onClick={onCancel}
      />
      <div
        className="relative rounded-xl p-0 w-full max-w-md shadow-2xl overflow-hidden animate-scale-in"
        style={{ background: "#ffffff" }}
      >
        {/* Close button */}
        <button
          onClick={onCancel}
          className="absolute top-4 right-4 p-1 rounded-md hover:bg-gray-100 transition-colors"
        >
          <X className="w-4 h-4" style={{ color: "var(--text-tertiary)" }} />
        </button>

        <div className="p-6 pb-0">
          {/* Icon */}
          <div
            className="w-12 h-12 rounded-full flex items-center justify-center mb-4"
            style={{ background: config.iconBg }}
          >
            <IconComponent
              className="w-6 h-6"
              style={{ color: config.iconColor }}
            />
          </div>

          {/* Title */}
          <h3
            className="text-lg font-semibold mb-2"
            style={{ color: "var(--text-primary)" }}
          >
            {title}
          </h3>

          {/* Message */}
          <p
            className="text-sm leading-relaxed"
            style={{ color: "var(--text-secondary)" }}
          >
            {message}
          </p>
        </div>

        {/* Actions */}
        <div
          className="flex gap-3 p-6 mt-4"
          style={{
            borderTop: "1px solid var(--border)",
            background: "var(--muted)",
          }}
        >
          <button
            onClick={onCancel}
            className="flex-1 px-4 py-2.5 rounded-lg text-sm font-medium transition-colors hover:bg-gray-100"
            style={{
              background: "#ffffff",
              border: "1px solid var(--border)",
              color: "var(--text-primary)",
            }}
          >
            {cancelLabel}
          </button>
          <button
            onClick={onConfirm}
            className="flex-1 px-4 py-2.5 rounded-lg text-sm font-medium transition-colors hover:opacity-90"
            style={{
              background: config.btnBg,
              color: config.btnText,
            }}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
