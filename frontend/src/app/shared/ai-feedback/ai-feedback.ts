import { Component, inject, input, signal } from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';

import { AgentService } from '../../core/services/agent.service';
import { CoachFeedback } from '../../core/models/agent.model';

/** Reusable "Get AI coaching feedback" widget for a saved session. */
@Component({
  selector: 'app-ai-feedback',
  imports: [],
  templateUrl: './ai-feedback.html',
  styleUrl: './ai-feedback.scss',
})
export class AiFeedback {
  readonly sessionId = input<number | null>(null);
  private agent = inject(AgentService);

  readonly loading = signal(false);
  readonly error = signal('');
  readonly feedback = signal<CoachFeedback | null>(null);

  async load(): Promise<void> {
    const id = this.sessionId();
    if (id == null || id < 0) {
      this.error.set('Session was not saved, so AI feedback is unavailable.');
      return;
    }
    this.loading.set(true);
    this.error.set('');
    try {
      this.feedback.set(await firstValueFrom(this.agent.feedback(id)));
    } catch (e) {
      const status = e instanceof HttpErrorResponse ? e.status : 0;
      this.error.set(
        status === 503
          ? 'AI coach is not configured (missing GROQ_API_KEY on the server).'
          : 'Could not get AI feedback right now. Please try again.',
      );
    } finally {
      this.loading.set(false);
    }
  }
}
