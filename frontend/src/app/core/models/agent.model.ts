/** Shapes returned by /api/agent/* (Groq Llama-generated). */

export interface CoachFeedback {
  summary: string;
  corrections: string[];
  tips: string[];
}

export interface PlanDay {
  day: string;
  focus: string;
  work?: string;
}

export interface TrainingPlan {
  // Marked optional: the LLM is prompted for all three, but we guard defensively.
  weekly_plan?: PlanDay[];
  focus_areas?: string[];
  progression_notes?: string;
}
