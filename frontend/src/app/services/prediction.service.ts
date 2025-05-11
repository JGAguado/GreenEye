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

  fetchPredictions(): void {
    this.http.get('/assets/mockResults_multimodel.json').subscribe(data => {
      this.predictionsSubject.next(data);
    });
  }

  setUploadedImageUrl(url: string): void {
    console.log('Setting uploaded image URL:', url); // Debugging log
    this.uploadedImageUrlSubject.next(url);
  }

  // Simulate prediction by uploading a file
  simulatePrediction(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);

    return this.http.post('/api/predict', formData).pipe(
      catchError(err => {
        console.error('API endpoint not available, falling back to mock data:', err);
        // Fallback to mock JSON if API fails
        return this.http.get('/assets/mockResults_multimodel.json');
      })
    );
  }

  // Update predictions manually
  updatePredictions(predictions: any): void {
    this.predictionsSubject.next(predictions);
  }
}