# GreenEye Backend API using FastAPI

import io
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from typing import List
from datetime import datetime

app = FastAPI(title="GreenEye Backend API")

ML_SERVER_URL = "http://ml-model-server:8501/v1/models/greeneye:predict"

# Helper function for image preprocessing
def preprocess_image(image: Image.Image) -> bytes:
    image = image.resize((224, 224))
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    return buffer.getvalue()

# Endpoint to upload a single or multiple images
@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded")

    predictions = []

    for file in files:
        try:
            # Read and preprocess image
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            processed_image = preprocess_image(image)

            # Send image to ML model server
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    ML_SERVER_URL,
                    files={"file": (file.filename, processed_image, "image/jpeg")}
                )

            if response.status_code != 200:
                raise HTTPException(status_code=502, detail="ML model server error")

            result = response.json()
            predictions.append(result)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    # Aggregation logic for multiple images (simple average of confidence scores)
    aggregated_result = aggregate_predictions(predictions)

    # Response with aggregated predictions and care instructions
    response_content = {
        "timestamp": datetime.utcnow().isoformat(),
        "aggregated_prediction": aggregated_result,
        "care_instructions": get_care_instructions(aggregated_result)
    }

    return JSONResponse(content=response_content)

# Aggregation function (example implementation)
def aggregate_predictions(predictions: List[dict]) -> dict:
    species_confidence = {}
    disease_confidence = {}

    for pred in predictions:
        species = pred.get('species')
        disease = pred.get('disease')

        species_confidence[species] = species_confidence.get(species, 0) + pred.get('species_confidence', 0)
        disease_confidence[disease] = disease_confidence.get(disease, 0) + pred.get('disease_confidence', 0)

    # Pick highest confidence
    final_species = max(species_confidence, key=species_confidence.get)
    final_disease = max(disease_confidence, key=disease_confidence.get)

    return {
        "species": final_species,
        "disease_status": final_disease,
        "species_confidence": species_confidence[final_species] / len(predictions),
        "disease_confidence": disease_confidence[final_disease] / len(predictions)
    }

# Function to retrieve care instructions (example stub)
def get_care_instructions(prediction: dict) -> str:
    species = prediction['species']
    disease_status = prediction['disease_status']

    if disease_status == "Healthy":
        return f"{species} is healthy. No action needed."
    else:
        return f"{species} detected with {disease_status}. Please apply recommended treatment."

# Root endpoint for quick health check
@app.get("/")
def read_root():
    return {"status": "GreenEye Backend API is running!"}
