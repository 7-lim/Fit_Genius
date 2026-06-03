/** End-of-session rollup returned by POST /api/pose/reset (and later persisted). */
export interface SessionSummary {
  exercise: string;
  reps: number;
  form_errors: Record<string, number>;
  avg_knee_angle: number;
  avg_hip_angle: number;
  avg_spine_angle: number;
  analyzed_frames: number;
}

/** Result of POST /api/pose/analyze-video. */
export interface VideoAnalysisResult extends SessionSummary {
  sampled_frames: number;
  detected_frames: number;
  warning?: string;
}

/** Body for POST /api/session/save. */
export interface SaveSessionPayload {
  exercise: string;
  reps: number;
  form_errors: Record<string, number>;
  duration_sec: number;
  avg_knee_angle?: number | null;
  avg_hip_angle?: number | null;
  avg_spine_angle?: number | null;
}

/** A persisted session row from GET /api/session/history. */
export interface SessionRecord {
  session_id: number;
  date: string;
  exercise: string;
  reps: number;
  duration_sec: number;
  form_errors: Record<string, number>;
  avg_knee_angle: number | null;
  avg_hip_angle: number | null;
  avg_spine_angle: number | null;
  top_errors: string[];
}
