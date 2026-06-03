import {
  AfterViewInit, Component, ElementRef, OnDestroy, OnInit, ViewChild, inject, signal,
} from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { firstValueFrom } from 'rxjs';

import { Exercise, PoseService } from '../../core/services/pose.service';
import { SessionService } from '../../core/services/session.service';
import { Landmark, PoseAnalysis } from '../../core/models/feedback.model';
import { SessionSummary } from '../../core/models/session.model';
import { AiFeedback } from '../../shared/ai-feedback/ai-feedback';

/** Frame send rate — 5fps is plenty for form detection (CLAUDE.md). */
const POLL_MS = 200;
const VIS_THRESHOLD = 0.4;

/** MediaPipe 33-point skeleton edges (body + limbs; face omitted for clarity). */
const POSE_CONNECTIONS: ReadonlyArray<[number, number]> = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],          // shoulders + arms
  [15, 17], [15, 19], [15, 21], [16, 18], [16, 20], [16, 22], // hands
  [11, 23], [12, 24], [23, 24],                               // torso
  [23, 25], [25, 27], [27, 29], [29, 31], [27, 31],           // left leg + foot
  [24, 26], [26, 28], [28, 30], [30, 32], [28, 32],           // right leg + foot
];

@Component({
  selector: 'app-workout',
  imports: [AiFeedback],
  templateUrl: './workout.html',
  styleUrl: './workout.scss',
})
export class Workout implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('video') private videoRef!: ElementRef<HTMLVideoElement>;
  @ViewChild('overlay') private overlayRef!: ElementRef<HTMLCanvasElement>;

  private route = inject(ActivatedRoute);
  private router = inject(Router);
  private pose = inject(PoseService);
  private sessions = inject(SessionService);

  exercise: Exercise = 'squat';
  private sessionId = 'web-' + Math.random().toString(36).slice(2);
  private startedAt = 0;

  // reactive UI state
  readonly status = signal<string>('starting');
  readonly phase = signal<string>('—');
  readonly formLabel = signal<string>('');
  readonly confidence = signal<number>(0);
  readonly reps = signal<number>(0);
  readonly feedback = signal<string>('');
  readonly errorFlagged = signal<boolean>(false);
  readonly cameraError = signal<string>('');
  readonly summary = signal<SessionSummary | null>(null);
  readonly savedId = signal<number | null>(null);

  private stream?: MediaStream;
  private timer?: ReturnType<typeof setInterval>;
  private inFlight = false;
  private readonly capture = document.createElement('canvas');

  ngOnInit(): void {
    const ex = this.route.snapshot.paramMap.get('exercise');
    this.exercise = ex === 'deadlift' ? 'deadlift' : 'squat';
  }

  async ngAfterViewInit(): Promise<void> {
    await this.startCamera();
  }

  ngOnDestroy(): void {
    this.teardown();
  }

  private async startCamera(): Promise<void> {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }, audio: false,
      });
      const video = this.videoRef.nativeElement;
      video.srcObject = this.stream;
      await video.play();
      this.startedAt = Date.now();
      this.status.set('warming_up');
      this.timer = setInterval(() => this.tick(), POLL_MS);
    } catch {
      this.cameraError.set(
        'Camera unavailable. Please allow camera access and reload.',
      );
    }
  }

  /** Capture one frame, send it for analysis, render the result. */
  private async tick(): Promise<void> {
    if (this.inFlight || this.summary()) return;          // skip if busy / ended
    const video = this.videoRef.nativeElement;
    if (!video.videoWidth) return;                        // not ready yet

    const cap = this.capture;
    cap.width = video.videoWidth;
    cap.height = video.videoHeight;
    cap.getContext('2d')!.drawImage(video, 0, 0, cap.width, cap.height);
    const frame = cap.toDataURL('image/jpeg', 0.6);

    this.inFlight = true;
    try {
      const res = await firstValueFrom(
        this.pose.analyze(frame, this.exercise, this.sessionId),
      );
      this.applyResult(res);
    } catch {
      /* transient network/decode error — drop this frame */
    } finally {
      this.inFlight = false;
    }
  }

  private applyResult(r: PoseAnalysis): void {
    this.status.set(r.status);
    this.reps.set(r.rep_count ?? this.reps());

    if (r.status === 'analyzing') {
      this.phase.set(r.phase ?? '—');
      this.formLabel.set(r.form_label ?? '');
      this.confidence.set(Math.round((r.confidence ?? 0) * 100));
      this.feedback.set(r.feedback ?? '');
      this.errorFlagged.set(!!r.error_flagged);
    } else {
      this.feedback.set(r.feedback ?? '');
      this.errorFlagged.set(false);
    }
    this.drawOverlay(r.landmarks);
  }

  private drawOverlay(landmarks?: Landmark[]): void {
    const video = this.videoRef.nativeElement;
    const cv = this.overlayRef.nativeElement;
    cv.width = video.clientWidth;
    cv.height = video.clientHeight;
    const ctx = cv.getContext('2d')!;
    ctx.clearRect(0, 0, cv.width, cv.height);
    if (!landmarks?.length) return;

    const pts = landmarks.map((l) => ({
      x: l.x * cv.width, y: l.y * cv.height, v: l.visibility,
    }));
    const color = this.errorFlagged() ? '#ef4444' : '#22c55e';

    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    for (const [a, b] of POSE_CONNECTIONS) {
      if (pts[a]?.v > VIS_THRESHOLD && pts[b]?.v > VIS_THRESHOLD) {
        ctx.beginPath();
        ctx.moveTo(pts[a].x, pts[a].y);
        ctx.lineTo(pts[b].x, pts[b].y);
        ctx.stroke();
      }
    }
    ctx.fillStyle = '#ffffff';
    for (const p of pts) {
      if (p.v > VIS_THRESHOLD) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  /** Stop the camera and fetch the session summary. */
  async stop(): Promise<void> {
    if (this.timer) clearInterval(this.timer);
    this.timer = undefined;
    this.stopCamera();
    try {
      const summary = await firstValueFrom(this.pose.reset(this.sessionId));
      this.summary.set(summary);
      if (summary) {
        await this.persist(summary);
      }
    } catch {
      this.summary.set(null);
    }
  }

  /** Save the finished session (best-effort; a failure shouldn't block the UI). */
  private async persist(summary: SessionSummary): Promise<void> {
    const duration_sec = Math.round((Date.now() - this.startedAt) / 1000);
    try {
      const id = await firstValueFrom(this.sessions.save({
        exercise: summary.exercise,
        reps: summary.reps,
        form_errors: summary.form_errors,
        duration_sec,
        avg_knee_angle: summary.avg_knee_angle,
        avg_hip_angle: summary.avg_hip_angle,
        avg_spine_angle: summary.avg_spine_angle,
      }));
      this.savedId.set(id);
    } catch {
      /* history save failed — non-critical */
    }
  }

  goHome(): void {
    this.router.navigate(['/home']);
  }

  formErrorList(): string[] {
    const errs = this.summary()?.form_errors ?? {};
    return Object.keys(errs);
  }

  private stopCamera(): void {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = undefined;
  }

  private teardown(): void {
    if (this.timer) clearInterval(this.timer);
    this.stopCamera();
  }
}
