import { Component, inject } from '@angular/core';
import { Router } from '@angular/router';

import { Exercise } from '../../core/services/pose.service';

@Component({
  selector: 'app-home',
  imports: [],
  templateUrl: './home.html',
  styleUrl: './home.scss',
})
export class Home {
  private router = inject(Router);

  live(exercise: Exercise): void {
    this.router.navigate(['/workout', exercise]);
  }

  upload(exercise: Exercise): void {
    this.router.navigate(['/upload', exercise]);
  }
}
