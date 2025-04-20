"""
model_inference.py

Module for loading Pl@ntNet-300K pretrained models and running inference on leaf images.
Includes a FastAPI endpoint for species prediction.
"""
import io
import os
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torch
import torchvision.transforms as transforms

# Import official model loader
from utils import load_model
from torchvision.models import resnet18

# Helper to load species model (official weights)
def load_species_model(use_gpu: bool = False) -> torch.nn.Module:
    """
    Initializes a ResNet18 model for 1081 classes and loads weights from a checkpoint.
    Args:
        use_gpu: whether to load weights onto GPU
    Returns:
        Model ready for inference (in eval mode)
    """
    filename = 'models/resnet18_weights_best_acc.tar'
    model = resnet18(num_classes=1081)
    load_model(model, filename=filename, use_gpu=use_gpu)
    model.eval()
    return model

# Image preprocessing pipeline
_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prediction function
def predict_species(model: torch.nn.Module, image: Image.Image, topk: int = 5) -> List[tuple]:
    """
    Runs species prediction on a PIL image and returns top-k predictions as (class_index, probability).
    """
    tensor = _transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        topk_probs, topk_indices = torch.topk(probs, topk)
    return [(int(idx), float(prob)) for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist())]

# FastAPI application
app = FastAPI(title="GreenEye Species Classifier")

@app.post('/predict/species/')
async def species_endpoint(
    file: UploadFile = File(...),
    topk: int = 5,
    use_gpu: bool = False
):
    """
    Upload an image to receive top-k species predictions.
    Query params:
      - use_gpu: whether to map model to GPU
    Returns JSON with 'predictions': [{'class_index': int, 'probability': float}, ...]
    """
    if file.content_type.split('/')[0] != 'image':
        raise HTTPException(status_code=400, detail='File is not an image.')
    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid image format.')

    model = load_species_model(use_gpu=use_gpu)
    preds = predict_species(model, img, topk)
    return {'predictions': [ {'class_index': idx, 'probability': prob} for idx, prob in preds ]}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
