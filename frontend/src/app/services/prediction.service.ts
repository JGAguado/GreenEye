import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, delay, of } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class PredictionService {
  predictions$ = new BehaviorSubject<any[]>([]);
  uploadedImageUrl$ = new BehaviorSubject<string | null>(null);

  constructor(private http: HttpClient) {}

  simulatePrediction(file: File) {
    // Store the image locally (in memory)
    const reader = new FileReader();
    reader.onload = () => {
      this.uploadedImageUrl$.next(reader.result as string);
    };
    reader.readAsDataURL(file);

    // Simulate delay + backend response
    return this.http.get<any>('assets/mockResults.json').pipe(delay(1000));
  }

  updatePredictions(predictions: any[]) {
    this.predictions$.next(predictions);
  }
}
