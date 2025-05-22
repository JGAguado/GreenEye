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
  supportedFormatsMessage = 'Supported formats: JPG, PNG. Max size: 10MB. Minimum resolution: 300x300px.';
  isInvalidFormat = false;

  constructor(
    private predictionService: PredictionService,
    private router: Router
  ) {}

  private validateAndSetFile(file: File): void {
    const validTypes = ['image/jpeg', 'image/png'];
    const maxFileSize = 10 * 1024 * 1024; // 10MB in bytes
    const minWidth = 300;
    const minHeight = 300;

    // Check file type
    if (!validTypes.includes(file.type)) {
      this.supportedFormatsMessage = 'Only JPG and PNG files are allowed.';
      this.isInvalidFormat = true;
      this.imageUrl = null;
      this.selectedFile = null;
      return;
    }

    // Check file size
    if (file.size > maxFileSize) {
      this.supportedFormatsMessage = 'File size must not exceed 10MB.';
      this.isInvalidFormat = true;
      this.imageUrl = null;
      this.selectedFile = null;
      return;
    }

    // Check image resolution
    const img = new Image();
    const reader = new FileReader();

    reader.onload = (event: ProgressEvent<FileReader>) => {
      if (event.target?.result) {
        img.src = event.target.result as string;
      }
    };

    img.onload = () => {
      if (img.width < minWidth || img.height < minHeight) {
        this.supportedFormatsMessage = `Image resolution must be at least ${minWidth}x${minHeight}px.`;
        this.isInvalidFormat = true;
        this.imageUrl = null;
        this.selectedFile = null;
      } else {
        // If all validations pass
        this.supportedFormatsMessage = '';
        this.isInvalidFormat = false;
        this.selectedFile = file;
        this.imageUrl = img.src; // Set the image preview
      }
    };

    img.onerror = () => {
      this.supportedFormatsMessage = 'Invalid image file.';
      this.isInvalidFormat = true;
      this.imageUrl = null;
      this.selectedFile = null;
    };

    reader.readAsDataURL(file);
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      this.validateAndSetFile(input.files[0]);
    }
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    if (event.dataTransfer?.files && event.dataTransfer.files[0]) {
      this.validateAndSetFile(event.dataTransfer.files[0]);
    }
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
  }

  removeImage(): void {
    this.imageUrl = null;
    this.selectedFile = null;
    this.supportedFormatsMessage = 'Supported formats: JPG, PNG. Max size: 10MB. Minimum resolution: 300x300px.';
    this.isInvalidFormat = false;
  }

  onSubmit(): void {
    if (!this.selectedFile) return;

    this.isLoading = true;

    // Create a URL for the uploaded file
    const fileUrl = URL.createObjectURL(this.selectedFile);
    this.predictionService.setUploadedImageUrl(fileUrl); // Set the uploaded image URL

    this.predictionService.predictWithBackend(this.selectedFile).subscribe({
      next: (res: any) => {
        // If your backend returns {pretrained_model: [...], custom_model: [...]}, update as needed
        this.predictionService.updatePredictions(res);
        this.isLoading = false;
        this.router.navigate(['/results']);
      },
      error: (err: any) => {
        this.isLoading = false;
        console.error('Backend error:', err);
      },
    });
  }
}