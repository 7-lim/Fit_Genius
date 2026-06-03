import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'home' },
  {
    path: 'home',
    loadComponent: () => import('./pages/home/home').then((m) => m.Home),
  },
  {
    path: 'workout/:exercise',
    loadComponent: () => import('./pages/workout/workout').then((m) => m.Workout),
  },
  {
    path: 'upload/:exercise',
    loadComponent: () => import('./pages/upload/upload').then((m) => m.Upload),
  },
  {
    path: 'history',
    loadComponent: () => import('./pages/history/history').then((m) => m.History),
  },
  { path: '**', redirectTo: 'home' },
];
