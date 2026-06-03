import { Component, inject, signal } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { firstValueFrom } from 'rxjs';

import { Exercise, PoseService } from '../../core/services/pose.service';
import { SessionService } from '../../core/services/session.service';
import { VideoAnalysisResult } from '../../core/models/session.model';
import { AiFeedback } from '../../shared/ai-feedback/ai-feedback';

@Component({
  selector: 'app-upload',
  imports: [AiFeedback],
  templateUrl: './upload.html',
  styleUrl: './upload.scss',
})
export class Upload {
  private route = inject(ActivatedRoute);
  private router = inject(Router);
  private pose = inject(PoseService);
  private sessions = inject(SessionService);

  exercise: Exercise =
    this.route.snapshot.paramMap.get('exercise') === 'deadlift' ? 'deadlift' : 'squat';

  readonly fileName = signal('');
  readonly loading = signal(false);
  readonly error = signal('');
  readonly result = signal<VideoAnalysisResult | null>(null);
  readonly savedId = signal<number | null>(null);
  private file?: File;

  onFile(event: Event): void {
    const input = event.target as HTMLInputElement;
    const f = input.files?.[0];
    if (f) {
      this.file = f;
      this.fileName.set(f.name);
      this.error.set('');
      this.result.set(null);
    }
  }

  async analyze(): Promise<void> {
    if (!this.file) {
      this.error.set('Choose a video file first.');
      return;
    }
    this.loading.set(true);
    this.error.set('');
    try {
      const res = await firstValueFrom(this.pose.analyzeVideo(this.file, this.exercise));
      this.result.set(res);
      try {
        const id = await firstValueFrom(this.sessions.save({
          exercise: this.exercise,
          reps: res.reps,
          form_errors: res.form_errors,
          duration_sec: Math.round(res.sampled_frames / 6),  // ~video length at 6fps
          avg_knee_angle: res.avg_knee_angle,
          avg_hip_angle: res.avg_hip_angle,
          avg_spine_angle: res.avg_spine_angle,
        }));
        this.savedId.set(id);
      } catch {
        /* history save failed — non-critical */
      }
    } catch {
      this.error.set(
        'Could not analyze the video. Try a shorter clip with your full body in frame.',
      );
    } finally {
      this.loading.set(false);
    }
  }

  formErrorList(): string[] {
    return Object.keys(this.result()?.form_errors ?? {});
  }

  reset(): void {
    this.file = undefined;
    this.fileName.set('');
    this.result.set(null);
    this.error.set('');
  }

  goHome(): void {
    this.router.navigate(['/home']);
  }
}
