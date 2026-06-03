import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, map } from 'rxjs';

import { ApiResponse } from '../models/feedback.model';
import { CoachFeedback, TrainingPlan } from '../models/agent.model';

/** All /api/agent/* calls (Groq Llama coaching). */
@Injectable({ providedIn: 'root' })
export class AgentService {
  private http = inject(HttpClient);

  /** Post-session coaching feedback for a saved session. */
  feedback(sessionId: number): Observable<CoachFeedback> {
    return this.http
      .post<ApiResponse<CoachFeedback>>('/api/agent/feedback', { session_id: sessionId })
      .pipe(map((res) => res.data as CoachFeedback));
  }

  /** A weekly training plan; history defaults to the user's stored sessions. */
  plan(userGoals?: string): Observable<TrainingPlan> {
    return this.http
      .post<ApiResponse<TrainingPlan>>('/api/agent/plan', { user_goals: userGoals })
      .pipe(map((res) => res.data as TrainingPlan));
  }
}
