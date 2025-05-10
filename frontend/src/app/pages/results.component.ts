import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatButtonModule } from '@angular/material/button';
import { HttpClientModule, HttpClient } from '@angular/common/http';
import { PredictionService } from '../services/prediction.service';

@Component({
  selector: 'app-results',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    HttpClientModule,
    MatCardModule,
    MatProgressBarModule,
    MatButtonModule
  ],
  templateUrl: './results.component.html',
  styleUrls: ['./results.component.scss']
})

export class ResultsComponent implements OnInit {
  predictions: any[] = [];
  uploadedImageUrl: string | null = null;

  constructor(private predictionService: PredictionService) {}

  ngOnInit(): void {
    this.predictionService.predictions$.subscribe(preds => {
      this.predictions = preds;
    });

    this.predictionService.uploadedImageUrl$.subscribe(url => {
      this.uploadedImageUrl = url;
    });
  }

  goBack() {
    window.history.back();
  }
}
