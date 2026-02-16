"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
} from "react";
import { createPortal } from "react-dom";
import { CheckCircle, XCircle, AlertTriangle, Info, X } from "lucide-react";

type ToastType = "success" | "error" | "warning" | "info";

interface Toast {
  id: string;
  type: ToastType;
  message: string;
  duration?: number;
}

interface ToastContextType {
  toast: {
    success: (message: string, duration?: number) => void;
    error: (message: string, duration?: number) => void;
    warning: (message: string, duration?: number) => void;
    info: (message: string, duration?: number) => void;
  };
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export function useToast() {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error("useToast must be used within a ToastProvider");
  }
  return context.toast;
}

const toastConfig: Record<
  ToastType,
  {
    icon: typeof CheckCircle;
    bg: string;
    border: string;
    iconColor: string;
    textColor: string;
  }
> = {
  success: {
    icon: CheckCircle,
    bg: "#f0fdf4",
    border: "#bbf7d0",
    iconColor: "var(--success)",
    textColor: "#166534",
  },
  error: {
    icon: XCircle,
    bg: "#fef2f2",
    border: "#fecaca",
    iconColor: "var(--danger)",
    textColor: "#991b1b",
  },
  warning: {
    icon: AlertTriangle,
    bg: "#fffbeb",
    border: "#fde68a",
    iconColor: "var(--warning)",
    textColor: "#92400e",
  },
  info: {
    icon: Info,
    bg: "#f0efff",
    border: "#c4b5fd",
    iconColor: "var(--accent)",
    textColor: "#3730a3",
  },
};

function ToastItem({
  toast,
  onRemove,
}: {
  toast: Toast;
  onRemove: (id: string) => void;
}) {
  const [isExiting, setIsExiting] = useState(false);
  const config = toastConfig[toast.type];
  const Icon = config.icon;

  useEffect(() => {
    const duration = toast.duration || 4000;
    const exitTimer = setTimeout(() => setIsExiting(true), duration - 300);
    const removeTimer = setTimeout(() => onRemove(toast.id), duration);
    return () => {
      clearTimeout(exitTimer);
      clearTimeout(removeTimer);
    };
  }, [toast.id, toast.duration, onRemove]);

  const handleClose = () => {
    setIsExiting(true);
    setTimeout(() => onRemove(toast.id), 300);
  };

  return (
    <div
      className={`flex items-start gap-3 px-4 py-3 rounded-lg shadow-lg border max-w-sm w-full transition-all duration-300 ${
        isExiting
          ? "opacity-0 translate-x-full"
          : "opacity-100 translate-x-0 animate-slide-in"
      }`}
      style={{
        background: config.bg,
        borderColor: config.border,
      }}
    >
      <Icon
        className="w-5 h-5 flex-shrink-0 mt-0.5"
        style={{ color: config.iconColor }}
      />
      <p
        className="text-sm font-medium flex-1 leading-snug"
        style={{ color: config.textColor }}
      >
        {toast.message}
      </p>
      <button
        onClick={handleClose}
        className="flex-shrink-0 rounded-md p-0.5 hover:opacity-70 transition-opacity"
      >
        <X className="w-4 h-4" style={{ color: config.textColor }} />
      </button>
    </div>
  );
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const addToast = useCallback(
    (type: ToastType, message: string, duration?: number) => {
      const id = `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      setToasts((prev) => [...prev, { id, type, message, duration }]);
    },
    [],
  );

  const toast = {
    success: (message: string, duration?: number) =>
      addToast("success", message, duration),
    error: (message: string, duration?: number) =>
      addToast("error", message, duration),
    warning: (message: string, duration?: number) =>
      addToast("warning", message, duration),
    info: (message: string, duration?: number) =>
      addToast("info", message, duration),
  };

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      {mounted &&
        createPortal(
          <div className="fixed top-4 right-4 z-[999999] flex flex-col gap-2">
            {toasts.map((t) => (
              <ToastItem key={t.id} toast={t} onRemove={removeToast} />
            ))}
          </div>,
          document.body,
        )}
    </ToastContext.Provider>
  );
}
