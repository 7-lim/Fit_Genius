import { DatePipe } from '@angular/common';
import { HttpErrorResponse } from '@angular/common/http';
import { Component, OnInit, inject, signal } from '@angular/core';
import { firstValueFrom } from 'rxjs';

import { SessionService } from '../../core/services/session.service';
import { AgentService } from '../../core/services/agent.service';
import { SessionRecord } from '../../core/models/session.model';
import { TrainingPlan } from '../../core/models/agent.model';

@Component({
  selector: 'app-history',
  imports: [DatePipe],
  templateUrl: './history.html',
  styleUrl: './history.scss',
})
export class History implements OnInit {
  private sessions = inject(SessionService);
  private agent = inject(AgentService);

  readonly loading = signal(true);
  readonly records = signal<SessionRecord[]>([]);
  readonly loadError = signal('');

  readonly planLoading = signal(false);
  readonly planError = signal('');
  readonly plan = signal<TrainingPlan | null>(null);

  async ngOnInit(): Promise<void> {
    try {
      this.records.set(await firstValueFrom(this.sessions.history()));
    } catch {
      this.loadError.set('Could not load history — is the backend running?');
    } finally {
      this.loading.set(false);
    }
  }

  async generatePlan(goals: string): Promise<void> {
    this.planLoading.set(true);
    this.planError.set('');
    try {
      this.plan.set(await firstValueFrom(this.agent.plan(goals.trim() || undefined)));
    } catch (e) {
      const status = e instanceof HttpErrorResponse ? e.status : 0;
      this.planError.set(
        status === 503
          ? 'AI coach is not configured (missing GROQ_API_KEY on the server).'
          : 'Could not generate a plan right now. Please try again.',
      );
    } finally {
      this.planLoading.set(false);
    }
  }

  fmtDuration(sec: number): string {
    const s = Math.max(0, Math.round(sec));
    const m = Math.floor(s / 60);
    return m ? `${m}m ${s % 60}s` : `${s}s`;
  }
}
