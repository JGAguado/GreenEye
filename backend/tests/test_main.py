import io
import json
from unittest.mock import MagicMock, patch

import pytest  # Import pytest to fix the NameError
import torch  # Import torch to fix the NameError
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, get_idx2species, predict_species

client = TestClient(app)


def test_hello_world():
    assert 1 + 1 == 2


def test_species_endpoint_valid_image():
    # Create a dummy image
    img = Image.new("RGB", (224, 224), color="white")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    # Mock the model and mappings
    with patch("app.main.get_model") as mock_get_model, patch(
        "app.main.get_idx2species"
    ) as mock_get_idx2species, patch(
        "app.main.predict_species"
    ) as mock_predict_species:
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        mock_get_idx2species.return_value = {0: "Species A", 1: "Species B"}
        mock_predict_species.return_value = [
            {"class_index": 0, "name": "Species A", "probability": 0.9},
            {"class_index": 1, "name": "Species B", "probability": 0.1},
        ]

        # Send POST request
        response = client.post(
            "/predict/species/",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"topk": 2, "use_gpu": False},
        )

        # Validate response
        assert response.status_code == 200
        assert response.json() == {
            "predictions": [
                {"class_index": 0, "name": "Species A", "probability": 0.9},
                {"class_index": 1, "name": "Species B", "probability": 0.1},
            ]
        }


def test_species_endpoint_invalid_file():
    # Send POST request with a non-image file
    response = client.post(
        "/predict/species/",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )

    # Validate response
    assert response.status_code == 400
    assert response.json() == {"detail": "File is not an image."}


def test_species_endpoint_invalid_image_format():
    # Send POST request with invalid image data
    response = client.post(
        "/predict/species/",
        files={"file": ("test.jpg", b"invalid image data", "image/jpeg")},
    )

    # Validate response
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid image format."}


def test_fetch_logs():
    # Mock the get_logs function
    with patch("app.main.get_logs") as mock_get_logs:
        mock_get_logs.return_value = [
            {"timestamp": "2025-05-03T12:00:00Z", "message": "Log 1"},
            {"timestamp": "2025-05-03T12:01:00Z", "message": "Log 2"},
        ]

        # Send GET request
        response = client.get("/logs/", params={"limit": 2})

        # Validate response
        assert response.status_code == 200
        assert response.json() == {
            "logs": [
                {"timestamp": "2025-05-03T12:00:00Z", "message": "Log 1"},
                {"timestamp": "2025-05-03T12:01:00Z", "message": "Log 2"},
            ]
        }


def test_get_idx2species():
    # Mock the JSON files
    with patch("builtins.open", create=True) as mock_open, patch(
        "os.path.exists", return_value=True
    ):
        mock_open.side_effect = [
            io.StringIO(json.dumps({"0": "species_0", "1": "species_1"})),
            io.StringIO(
                json.dumps({"species_0": "Species A", "species_1": "Species B"})
            ),
        ]

        idx2species = get_idx2species()
        assert idx2species == {0: "Species A", 1: "Species B"}


def test_predict_species():
    # Create a dummy image
    img = Image.new("RGB", (224, 224), color="white")

    # Mock the model and mappings
    mock_model = MagicMock()
    # Use logits that will result in probabilities closer to the expected values
    mock_model.return_value = torch.tensor(
        [[2.0, 1.0]]
    )  # Softmax will produce ~[0.731, 0.269]
    mock_idx2species = {0: "Species A", 1: "Species B"}

    # Call the function
    results = predict_species(mock_model, img, topk=2, idx2species=mock_idx2species)

    # Validate results
    assert results == [
        {
            "class_index": 0,
            "name": "Species A",
            "probability": pytest.approx(0.731, rel=1e-2),
        },
        {
            "class_index": 1,
            "name": "Species B",
            "probability": pytest.approx(0.269, rel=1e-2),
        },
    ]
