import { create } from "zustand";

interface AnalysisFile {
  file: File;
  name: string;
  size: number;
}

interface AnalysisState {
  files: AnalysisFile[];
  summarizationModel: "gpt4" | "local";
  setFiles: (files: AnalysisFile[]) => void;
  setSummarizationModel: (model: "gpt4" | "local") => void;
  clearFiles: () => void;
}

export const useAnalysisStore = create<AnalysisState>()((set) => ({
  files: [],
  summarizationModel: "gpt4",
  setFiles: (files) => set({ files }),
  setSummarizationModel: (model) => set({ summarizationModel: model }),
  clearFiles: () => set({ files: [], summarizationModel: "gpt4" }),
}));
