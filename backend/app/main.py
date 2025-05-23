"""
main.py

Module for loading Pl@ntNet-300K pretrained models and running inference on leaf images.
Includes a FastAPI endpoint for species prediction.
"""

import io
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision.models import resnet18
import timm

from app.logger import get_logs, log_request

# Import official model loader
from app.utils import load_model, get_model

# Paths to mapping files
CLASS_IDX_TO_SPECIES_ID_PATH = "models/class_idx_to_species_id.json"
SPECIES_ID_TO_NAME_PATH = "models/plantnet300K_species_id_2_name.json"

from functools import lru_cache

from fastapi.middleware.cors import CORSMiddleware


@lru_cache()
def get_pretrained_model(use_gpu=False):
    return load_species_model(use_gpu=use_gpu)


@lru_cache()
def get_custom_model(use_gpu=False):
    return load_custom_model(use_gpu=use_gpu)


# Helper to load species model (official weights)
def load_species_model(use_gpu: bool = False) -> torch.nn.Module:
    """
    Initializes a ResNet18 model for 1081 classes and loads weights from a checkpoint.
    Args:
        use_gpu: whether to load weights onto GPU
    Returns:
        Model ready for inference (in eval mode)
    """
    filename = "models/resnet18_weights_best_acc.tar"
    model = resnet18(num_classes=1081)
    load_model(model, filename=filename, use_gpu=use_gpu)
    model.eval()
    return model


# Helper to load custom model
def load_custom_model(use_gpu: bool = False) -> torch.nn.Module:
    """
    Initializes and loads the custom EfficientNet-B4 model.
    Args:
        use_gpu: whether to load weights onto GPU
    Returns:
        Model ready for inference (in eval mode)
    """
    filename = "models/efficientnet_b4_plantnet.pth"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Custom model weights not found at {filename}")
    
    # Initialize EfficientNet-B4 model using torchvision
    model = torchvision.models.efficientnet_b4(pretrained=False, num_classes=1081)
    load_model(model, filename=filename, use_gpu=use_gpu)
    model.eval()
    return model


# Load mappings once
def get_idx2species(
    idx2id_path: str = CLASS_IDX_TO_SPECIES_ID_PATH,
    id2name_path: str = SPECIES_ID_TO_NAME_PATH,
) -> Dict[int, str]:
    """
    Loads two JSON mappings:
      1. class index -> species id
      2. species id -> species scientific name
    Returns a dict mapping int class index to species name.
    """
    # Load class_idx_to_species_id.json
    if not os.path.exists(idx2id_path):
        raise FileNotFoundError(f"Mapping file not found: {idx2id_path}")
    with open(idx2id_path, "r") as f:
        raw_idx2id = json.load(f)
    # Load species_id to name mapping
    if not os.path.exists(id2name_path):
        raise FileNotFoundError(f"Mapping file not found: {id2name_path}")
    with open(id2name_path, "r") as f:
        raw_id2name = json.load(f)
    # Build combined mapping
    idx2species: Dict[int, str] = {}
    for idx_str, species_id in raw_idx2id.items():
        idx = int(idx_str)
        name = raw_id2name.get(species_id, "Unknown")
        idx2species[idx] = name
    return idx2species


# Image preprocessing pipeline
_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Prediction function
def predict_species(
    model: torch.nn.Module,
    image: Image.Image,
    topk: int = 5,
    idx2species: Dict[int, str] = None,
) -> List[Dict]:
    """
    Runs species prediction on a PIL image and returns top-k predictions.
    Returns a list of dicts: {'class_index': int, 'name': str, 'probability': float}
    """
    if idx2species is None:
        idx2species = get_idx2species()
    tensor = _transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        topk_probs, topk_indices = torch.topk(probs, topk)

    results: List[Dict] = []
    for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
        results.append(
            {
                "class_index": int(idx),
                "name": idx2species.get(int(idx), "Unknown"),
                "probability": float(prob),
            }
        )
    return results


# FastAPI application
app = FastAPI(title="GreenEye Species Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Angular dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/species/")
async def species_endpoint(
    file: UploadFile = File(...),
    topk: int = 5,
    use_gpu: bool = False,
):
    """
    Upload an image to receive top-k species predictions from both pretrained and custom models.
    Query params:
      - use_gpu: whether to map model to GPU
    Returns JSON with predictions from both models:
    {
      "pretrained_model": [
        {
          "class_index": int,
          "name": str,
          "probability": float
        },
        ...
      ],
      "custom_model": [
        {
          "class_index": int,
          "name": str,
          "probability": float
        },
        ...
      ]
    }
    """
    # Validate file type
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="File is not an image.")
    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image format.")

    # Load models and mappings
    pretrained_model = get_pretrained_model(use_gpu=use_gpu)
    idx2species = get_idx2species()
    
    # Get predictions from pretrained model
    pretrained_preds = predict_species(pretrained_model, img, topk, idx2species)
    
    # Initialize response with pretrained model predictions
    response = {
        "pretrained_model": pretrained_preds,
        "custom_model": None
    }
    
    try:
        # Try to get predictions from custom model
        custom_model = get_custom_model(use_gpu=use_gpu)
        custom_preds = predict_species(custom_model, img, topk, idx2species)
        response["custom_model"] = custom_preds
    except FileNotFoundError:
        # If custom model is not available, set custom_model to None
        pass
    
    log_request(
        {
            "timestamp": datetime.utcnow().isoformat(),
            "image_filename": file.filename,
            "topk": topk,
            "results": response,
        }
    )
    return response


@app.get("/logs/")
def fetch_logs(limit: int = 100):
    return JSONResponse(content={"logs": get_logs(limit)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
