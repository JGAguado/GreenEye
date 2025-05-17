import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, of } from 'rxjs';
import { catchError } from 'rxjs/operators';

@Injectable({
  providedIn: 'root'
})
export class PredictionService {
  private predictionsSubject = new BehaviorSubject<any>(null);
  predictions$ = this.predictionsSubject.asObservable();

  private uploadedImageUrlSubject = new BehaviorSubject<string | null>(null);
  uploadedImageUrl$ = this.uploadedImageUrlSubject.asObservable();

  constructor(private http: HttpClient) {}

  fetchPredictions(file: File): void {
    const formData = new FormData();
    formData.append('file', file);

    const url = 'http://localhost:8000/predict/species/?topk=5&use_gpu=false';

    this.http.post(url, formData).subscribe({
      next: (data) => {
        this.predictionsSubject.next(data);
      },
      error: (err) => {
        console.error('Error fetching predictions from backend:', err);
        throw new Error('Failed to get prediction from backend.');
      }
    });
  }

  setUploadedImageUrl(url: string): void {
    console.log('Setting uploaded image URL:', url); // Debugging log
    this.uploadedImageUrlSubject.next(url);
  }

  // Use mockup JSON for simulation
  simulatePrediction(file: File): Observable<any> {
    return this.http.get('/assets/mockResults_multimodel.json');
  }

  // Real connection to backend, throws error if no JSON is received
  predictWithBackend(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);

    const url = 'http://localhost:8000/predict/species/?topk=5&use_gpu=false';

    return this.http.post(url, formData).pipe(
      // If backend does not return JSON, throw error
      catchError(err => {
        console.error('Error connecting to backend:', err);
        throw new Error('Failed to get prediction from backend.');
      })
    );
  }

  // Update predictions manually
  updatePredictions(predictions: any): void {
    this.predictionsSubject.next(predictions);
  }
}