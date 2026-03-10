import { create } from "zustand";
import api from "@/lib/api";

interface User {
  id: number;
  username: string;
  email: string;
  first_name: string;
  last_name: string;
  phone?: string;
  role: "admin" | "head_of_department" | "agent";
  company?: number;
  company_name?: string;
}

interface AuthState {
  user: User | null;
  isHydrating: boolean;
  setAuth: (user: User) => void;
  clearAuth: () => void;
  isAuthenticated: () => boolean;
  hydrateFromServer: () => Promise<void>;
}

export const useAuthStore = create<AuthState>()((set, get) => ({
  user: null,
  isHydrating: true,
  setAuth: (user) => {
    set({ user, isHydrating: false });
  },
  clearAuth: () => {
    set({ user: null, isHydrating: false });
  },
  isAuthenticated: () => !!get().user,
  hydrateFromServer: async () => {
    try {
      const response = await api.get("/users/me/");
      set({ user: response.data, isHydrating: false });
    } catch {
      set({ user: null, isHydrating: false });
    }
  },
}));
