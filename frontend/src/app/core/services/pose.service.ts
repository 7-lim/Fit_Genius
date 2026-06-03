import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, map } from 'rxjs';

import { ApiResponse, PoseAnalysis } from '../models/feedback.model';
import { SessionSummary, VideoAnalysisResult } from '../models/session.model';

export type Exercise = 'squat' | 'deadlift';

/** All /api/pose/* calls. Components never touch HttpClient directly. */
@Injectable({ providedIn: 'root' })
export class PoseService {
  private http = inject(HttpClient);

  /** Send one webcam frame (base64 jpeg) for live analysis. */
  analyze(frame: string, exercise: Exercise, sessionId: string): Observable<PoseAnalysis> {
    return this.http
      .post<ApiResponse<PoseAnalysis>>('/api/pose/analyze', {
        frame, exercise, session_id: sessionId,
      })
      .pipe(map((res) => res.data as PoseAnalysis));
  }

  /** Upload a whole video for offline analysis. */
  analyzeVideo(file: File, exercise: Exercise): Observable<VideoAnalysisResult> {
    const form = new FormData();
    form.append('video', file);
    form.append('exercise', exercise);
    return this.http
      .post<ApiResponse<VideoAnalysisResult>>('/api/pose/analyze-video', form)
      .pipe(map((res) => res.data as VideoAnalysisResult));
  }

  /** End the live session; returns its final summary. */
  reset(sessionId: string): Observable<SessionSummary | null> {
    return this.http
      .post<ApiResponse<{ summary: SessionSummary | null }>>('/api/pose/reset', {
        session_id: sessionId,
      })
      .pipe(map((res) => res.data?.summary ?? null));
  }
}
