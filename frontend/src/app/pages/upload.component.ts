import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { DomSanitizer, SafeUrl } from '@angular/platform-browser';

@Component({
  selector: 'app-upload',
  standalone: true,
  imports: [CommonModule, MatButtonModule, MatIconModule],
  templateUrl: './upload.component.html',
  styleUrls: ['./upload.component.scss'],
})
export class UploadComponent {
  imageUrl: SafeUrl | null = null;
  isLoading = false;
  supportedFormatsMessage = 'Supported formats: JPG, PNG';
  isInvalidFormat = false; // Tracks if the format is invalid

  constructor(private sanitizer: DomSanitizer) {}

  private validateAndSetFile(file: File): void {
    const validTypes = ['image/jpeg', 'image/png'];

    if (!validTypes.includes(file.type)) {
      this.supportedFormatsMessage = 'Only JPG and PNG files are allowed.';
      this.isInvalidFormat = true;
      this.imageUrl = null;
      return;
    }

    this.supportedFormatsMessage = '';
    this.isInvalidFormat = false;
    const reader = new FileReader();

    reader.onload = () => {
      this.imageUrl = reader.result as string; // Set the image URL for preview
    };

    reader.readAsDataURL(file); // Read the file as a data URL
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;

      if (input.files && input.files[0]) {
        const file = input.files[0];
        this.validateAndSetFile(file);
      }
  }

  onDrop(event: DragEvent) {
    event.preventDefault();
    
    if (event.dataTransfer?.files && event.dataTransfer.files[0]) {
      const file = event.dataTransfer.files[0];
      this.validateAndSetFile(file);
    }
  }

  removeImage() {
    this.imageUrl = null;
    this.supportedFormatsMessage = 'Supported formats: JPG, PNG'; // Reset to default message
    this.isInvalidFormat = false; // Reset invalid format flag
  }

  submit() {
    this.isLoading = true;
    setTimeout(() => {
      this.isLoading = false;
      alert('Mock submission complete!');
    }, 2000);
  }

  onDragOver(event: DragEvent) {
    event.preventDefault();
  }
}
