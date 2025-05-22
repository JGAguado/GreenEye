import { Routes } from '@angular/router';
import { UploadComponent } from './pages/upload.component';
import { ResultsComponent } from './pages/results.component';

export const routes: Routes = [
  { path: '', component: UploadComponent },
  { path: 'results', component: ResultsComponent },
  { path: '**', redirectTo: '' }, // catch-all fallback to upload page
];
