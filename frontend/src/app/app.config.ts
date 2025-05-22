import { provideRouter } from '@angular/router';
import { UploadComponent } from './pages/upload.component';

export const appConfig = {
  providers: [
    provideRouter([
      { path: '', component: UploadComponent },
    ])
  ]
};
