/** Shapes returned by /api/pose/* (mirrors the Flask response envelope). */

export interface Landmark {
  x: number;          // normalized [0,1]
  y: number;          // normalized [0,1]
  z: number;
  visibility: number; // [0,1]
}

export interface Angles {
  knee: number;
  hip: number;
  spine: number;
}

export type AnalysisStatus = 'warming_up' | 'analyzing' | 'no_pose';

/** Result of POST /api/pose/analyze (the `data` field). */
export interface PoseAnalysis {
  status: AnalysisStatus;
  rep_count: number;
  frames_needed?: number;        // warming_up only
  phase?: 'up' | 'down';
  form_class?: string;
  form_label?: string;
  confidence?: number;           // 0..1
  angles?: Angles;
  error_flagged?: boolean;
  feedback?: string;
  landmarks?: Landmark[];        // 33 points for the overlay
}

/** Standard backend envelope: { data, error }. */
export interface ApiResponse<T> {
  data: T | null;
  error: string | null;
}
