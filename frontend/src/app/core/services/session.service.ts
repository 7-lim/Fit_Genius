import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, map } from 'rxjs';

import { ApiResponse } from '../models/feedback.model';
import { SaveSessionPayload, SessionRecord } from '../models/session.model';

/** All /api/session/* calls. */
@Injectable({ providedIn: 'root' })
export class SessionService {
  private http = inject(HttpClient);

  /** Persist a finished session; resolves to the new session id. */
  save(payload: SaveSessionPayload): Observable<number> {
    return this.http
      .post<ApiResponse<{ session_id: number }>>('/api/session/save', payload)
      .pipe(map((res) => res.data?.session_id ?? -1));
  }

  /** Past sessions, most recent first. */
  history(): Observable<SessionRecord[]> {
    return this.http
      .get<ApiResponse<SessionRecord[]>>('/api/session/history')
      .pipe(map((res) => res.data ?? []));
  }
}
