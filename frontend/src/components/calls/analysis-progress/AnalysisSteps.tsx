import React from "react";
import {
  Upload,
  Mic,
  FileAudio,
  Heart,
  Brain,
  MessageSquare,
  Lightbulb,
  CheckCircle,
} from "lucide-react";
import { AnalysisStep } from "./analysisConstants";

type StepDef = Omit<AnalysisStep, "status">;

export const ANALYSIS_STEPS: StepDef[] = [
  {
    id: "upload",
    label: "Uploading File",
    description: "Sending audio file to the server...",
    icon: <Upload className="w-5 h-5" />,
  },
  {
    id: "transcribe",
    label: "Transcribing Audio",
    description: "Converting speech to text using Whisper AI...",
    icon: <Mic className="w-5 h-5" />,
  },
  {
    id: "summarize",
    label: "Generating Summary",
    description: "Creating a concise summary of the conversation...",
    icon: <FileAudio className="w-5 h-5" />,
  },
  {
    id: "emotion",
    label: "Analyzing Emotions",
    description: "Detecting emotional patterns throughout the call...",
    icon: <Heart className="w-5 h-5" />,
  },
  {
    id: "behavior",
    label: "Analyzing Behavior",
    description: "Evaluating communication patterns and techniques...",
    icon: <Brain className="w-5 h-5" />,
  },
  {
    id: "topics",
    label: "Extracting Topics",
    description: "Identifying key topics and themes discussed...",
    icon: <MessageSquare className="w-5 h-5" />,
  },
  {
    id: "coaching",
    label: "Generating Coaching Tips",
    description: "Creating personalized improvement suggestions...",
    icon: <Lightbulb className="w-5 h-5" />,
  },
  {
    id: "complete",
    label: "Analysis Complete",
    description: "Your call report is ready to view!",
    icon: <CheckCircle className="w-5 h-5" />,
  },
];
