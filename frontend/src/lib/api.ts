import axios from "axios";

export interface BulkUploadCallResult {
  external_id?: string;
  filename: string;
  status?: "pending";
  error?: string;
}

export interface BulkUploadResponse {
  total: number;
  imported: number;
  failed: number;
  calls: BulkUploadCallResult[];
}

export interface QueueStatus {
  awaiting_import: number;
  in_queue: number;
  processing: number;
  completed_today: number;
  failed_today: number;
}

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

// Create axios instance — withCredentials sends httpOnly cookies automatically
export const api = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: true,
  headers: {
    "Content-Type": "application/json",
  },
});

// Request interceptor — strip Content-Type for multipart uploads
api.interceptors.request.use(
  (config) => {
    if (config.data instanceof FormData) {
      delete config.headers["Content-Type"];
    }
    return config;
  },
  (error) => Promise.reject(error),
);

// Guard against multiple simultaneous logout redirects
let isLoggingOut = false;

function forceLogout() {
  if (isLoggingOut) return;
  isLoggingOut = true;
  window.location.href = "/login";
}

// Response interceptor — refresh expired access token via cookie-based refresh
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        // POST to refresh endpoint; the refresh_token cookie is sent automatically.
        // The server sets a new access_token cookie in the response.
        await axios.post(
          `${API_BASE_URL}/users/token/refresh/`,
          {},
          { withCredentials: true },
        );

        // Retry the original request — the new access_token cookie is now set
        return api(originalRequest);
      } catch {
        forceLogout();
        return Promise.reject(error);
      }
    }

    return Promise.reject(error);
  },
);

// Auth API functions
export const authApi = {
  login: (username: string, password: string) =>
    api.post("/users/login/", { username, password }),

  getCurrentUser: () => api.get("/users/me/"),

  updateProfile: (data: {
    first_name?: string;
    last_name?: string;
    email?: string;
    phone?: string;
  }) => api.patch("/users/me/", data),

  logout: () => api.post("/users/logout/"),
};

// Calls API functions
export const callsApi = {
  uploadCall: (formData: FormData) => api.post("/calls/upload/", formData),

  getMyCalls: () => api.get("/calls/my-calls/"),

  getCallDetail: (id: number) => api.get(`/calls/${id}/`),

  getCallStatus: (id: number) => api.get(`/calls/${id}/status/`),

  getQueueStatus: () => api.get("/calls/queue-status/"),
};

// Reports API functions
export const reportsApi = {
  generateReport: (
    agentId: number,
    reportType: "week" | "month",
    startDate?: string,
    endDate?: string,
  ) =>
    api.post("/reports/generate/", {
      agent_id: agentId,
      report_type: reportType,
      start_date: startDate,
      end_date: endDate,
    }),

  getMyReports: () => api.get("/reports/my-reports/"),

  getAgentReports: (agentId?: number) =>
    agentId ? api.get(`/reports/agent/${agentId}/`) : api.get("/reports/all/"),

  getReportDetail: (reportId: number) => api.get(`/reports/${reportId}/`),
};

// Users API functions
export const usersApi = {
  getAllUsers: (role?: string) => api.get("/users/all/", { params: { role } }),

  getSubordinates: () => api.get("/users/subordinates/"),

  createAgent: (data: {
    username: string;
    email: string;
    password: string;
    first_name?: string;
    last_name?: string;
    phone?: string;
    company?: number | null;
    reporting_to?: number | null;
  }) => api.post("/users/create-agent/", data),

  createHead: (data: {
    username: string;
    email: string;
    password: string;
    first_name?: string;
    last_name?: string;
    phone?: string;
    company?: number | null;
    reporting_to?: number | null;
  }) => api.post("/users/create-head/", data),

  getUserDetail: (id: number) => api.get(`/users/${id}/`),

  updateUser: (id: number, data: any) => api.patch(`/users/${id}/`, data),

  deleteUser: (id: number) => api.delete(`/users/${id}/`),
};

// Companies API functions
export const companiesApi = {
  getAllCompanies: () => api.get("/companies/all/"),

  createCompany: (data: { name: string; keywords?: string[] }) =>
    api.post("/companies/create/", data),

  getCompanyDetail: (id: number) => api.get(`/companies/${id}/`),

  updateCompany: (id: number, data: any) =>
    api.patch(`/companies/${id}/`, data),

  deleteCompany: (id: number) => api.delete(`/companies/${id}/`),

  getCompanyEmployees: (id: number) => api.get(`/companies/${id}/employees/`),

  assignHead: (companyId: number, headId: number) =>
    api.post("/companies/assign-head/", {
      company_id: companyId,
      head_id: headId,
    }),
};

// Mock Call Center API — separate axios instance authenticated with X-API-Key.
// This simulates an external call center platform. The API key is set via
// NEXT_PUBLIC_MOCK_API_KEY (defaults to 'dev-secret-key' for local development).
const MOCK_API_KEY =
  process.env.NEXT_PUBLIC_MOCK_API_KEY || "dev-secret-key";

const MOCK_BASE_URL =
  (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api") +
  "/mock-callcenter";

export const mockApi = axios.create({
  baseURL: MOCK_BASE_URL,
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": MOCK_API_KEY,
  },
});

// Strip Content-Type for multipart uploads
mockApi.interceptors.request.use((config) => {
  if (config.data instanceof FormData) {
    delete config.headers["Content-Type"];
  }
  return config;
});

export const mockCallCenterApi = {
  // Upload a new call recording to the mock call center
  uploadCall: (formData: FormData) =>
    mockApi.post("/calls/upload/", formData),

  // Upload multiple call recordings in a single request
  bulkUploadCalls: (formData: FormData) =>
    mockApi.post("/calls/bulk-upload/", formData),

  // Get all calls (optionally filter by analyzed status)
  getCalls: (analyzed?: boolean) =>
    mockApi.get("/calls/", {
      params: analyzed !== undefined ? { analyzed: String(analyzed) } : {},
    }),

  // Get unanalyzed calls only
  getUnanalyzedCalls: () =>
    mockApi.get("/calls/", { params: { analyzed: "false" } }),
};

export default api;
