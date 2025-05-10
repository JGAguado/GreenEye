import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { DomSanitizer, SafeUrl } from '@angular/platform-browser';
import { PredictionService } from '../services/prediction.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-upload',
  standalone: true,
  imports: [CommonModule, MatButtonModule, MatIconModule],
  templateUrl: './upload.component.html',
  styleUrls: ['./upload.component.scss'],
})

export class UploadComponent {
  imageUrl: SafeUrl | null = null;
  selectedFile: File | null = null;
  isLoading = false;
  supportedFormatsMessage = 'Supported formats: JPG, PNG';
  isInvalidFormat = false;

  constructor(
    private predictionService: PredictionService,
    private router: Router
  ) {}

  private validateAndSetFile(file: File): void {
    const validTypes = ['image/jpeg', 'image/png'];

    if (!validTypes.includes(file.type)) {
      this.supportedFormatsMessage = 'Only JPG and PNG files are allowed.';
      this.isInvalidFormat = true;
      this.imageUrl = null;
      this.selectedFile = null;
      return;
    }

    this.supportedFormatsMessage = '';
    this.isInvalidFormat = false;
    this.selectedFile = file;

    const reader = new FileReader();
    reader.onload = () => {
      this.imageUrl = reader.result as string;
    };
    reader.readAsDataURL(file);
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      this.validateAndSetFile(input.files[0]);
    }
  }

  onDrop(event: DragEvent) {
    event.preventDefault();
    if (event.dataTransfer?.files && event.dataTransfer.files[0]) {
      this.validateAndSetFile(event.dataTransfer.files[0]);
    }
  }

  onDragOver(event: DragEvent) {
    event.preventDefault();
  }

  removeImage() {
    this.imageUrl = null;
    this.selectedFile = null;
    this.supportedFormatsMessage = 'Supported formats: JPG, PNG';
    this.isInvalidFormat = false;
  }

  onSubmit() {
    if (!this.selectedFile) return;

    this.isLoading = true;
    this.predictionService.simulatePrediction(this.selectedFile).subscribe({
      next: res => {
        this.predictionService.updatePredictions(res.predictions);
        this.isLoading = false;
        this.router.navigate(['/results']);
      },
      error: err => {
        this.isLoading = false;
        console.error('Simulation error:', err);
      }
    });
  }
}
