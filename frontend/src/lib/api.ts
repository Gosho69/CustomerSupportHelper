import axios from "axios";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

// Create axios instance with default config
export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});
// If there's an access token in localStorage on load, set the default Authorization header
if (typeof window !== "undefined") {
  const token = localStorage.getItem("access_token");
  if (token) {
    api.defaults.headers.common.Authorization = `Bearer ${token}`;
  }
}
// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    if (typeof window !== "undefined") {
      // Try to get token from localStorage (set by auth store)
      const token = localStorage.getItem("access_token");
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
    }

    // If the data is FormData, delete Content-Type to let the browser set it
    // with the proper boundary for multipart/form-data
    if (config.data instanceof FormData) {
      delete config.headers["Content-Type"];
    }

    return config;
  },
  (error) => {
    return Promise.reject(error);
  },
);

// Response interceptor to handle token refresh
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    // If error is 401 and we haven't retried yet
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        const refreshToken = localStorage.getItem("refresh_token");
        if (refreshToken) {
          const response = await axios.post(
            `${API_BASE_URL}/users/token/refresh/`,
            {
              refresh: refreshToken,
            },
          );

          const { access } = response.data;
          localStorage.setItem("access_token", access);

          // Immediately update axios default header as well
          api.defaults.headers.common.Authorization = `Bearer ${access}`;

          // Retry original request with new token
          originalRequest.headers.Authorization = `Bearer ${access}`;
          return api(originalRequest);
        }
      } catch (refreshError) {
        // Refresh failed, logout user
        localStorage.removeItem("access_token");
        localStorage.removeItem("refresh_token");
        localStorage.removeItem("user");
        window.location.href = "/login";
        return Promise.reject(refreshError);
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

  logout: () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    localStorage.removeItem("user");
  },
};

// Calls API functions
export const callsApi = {
  uploadCall: (formData: FormData) => api.post("/calls/upload/", formData),

  getMyCalls: () => api.get("/calls/my-calls/"),

  getCallDetail: (id: number) => api.get(`/calls/${id}/`),
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

export default api;
