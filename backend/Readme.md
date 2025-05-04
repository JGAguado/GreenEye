# GreenEye Backend

This is the backend service for the GreenEye application, built with FastAPI.

## 📦 Dependencies

The project uses the following dependencies:

```
fastapi==0.110.0
uvicorn[standard]==0.29.0
httpx==0.27.0
pillow==10.3.0
python-multipart==0.0.9
timm==0.9.12
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
python-dotenv>=1.0.0
bcrypt==3.2.0
passlib[bcrypt]>=1.7.4
python-jose>=3.3.0
```

### Package Descriptions

- **fastapi:** Modern web framework for building APIs with Python
- **uvicorn:** ASGI server for running FastAPI apps
- **httpx:** Async HTTP client
- **pillow:** Image file processing
- **python-multipart:** File upload support
- **timm:** PyTorch image model utilities
- **torch & torchvision:** Core ML framework and vision utilities
- **numpy:** Array computing
- **python-dotenv:** Environment variable loader
- **bcrypt:** Password hashing backend
- **passlib:** High-level password hashing API
- **python-jose:** JSON Web Token (JWT) management

---

## 🛠️ Setup

### Local Development

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Unix/macOS
# or
venv\Scripts\activate      # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

### Docker Setup

1. Build the Docker image:

```bash
docker build -t greeneye-backend .
```

2. Run the container:

```bash
docker run -p 8000:8000 greeneye-backend
```

The service will be available at `http://localhost:8000`

---

## 🚀 Running the Application

### Local (with auto-reload):

```bash
uvicorn main.main:app --reload
```

---

## 🧪 Running Unit Tests

### Running Tests:

To run unit tests, use:

```bash
pytest
```

### Checking Test Coverage:

To check test coverage, use:

```bash
cd backend
pytest --cov=app
```

To generate an HTML coverage report, use:

```bash
cd backend
pytest --cov=app --cov-report=html
```

The HTML report will be available in the `htmlcov/` directory.

---

## 🌿 Inference API

### POST `/predict/species/`

This endpoint runs a model prediction on a plant image.

- Requires: **Authorization: Bearer <token>**
- Input: Image file (jpeg/png) as `file`
- Output: Prediction response

#### Example:

```bash
curl -X POST "http://localhost:8000/predict/species/" \
  -F "file=@leaf.jpg"
```

---

## 📁 Project Structure

```
.
├── app
│   ├── logger.py                 # Logging utilities
│   ├── main.py                   # FastAPI app with inference logic
│   └── utils.py                  # Helper functions and model utilities
├── Dockerfile                    # Docker configuration for containerization
├── logs
│   └── requests.jsonl            # Log file for API requests
├── models
│   ├── class_idx_to_species_id.json  # Mapping of class indices to species IDs
│   ├── plantnet300K_species_id_2_name.json  # Mapping of species IDs to names
│   └── resnet18_weights_best_acc.tar  # Pretrained model weights
├── noxfile.py                    # Nox configuration for automation
├── pytest.ini                    # Pytest configuration file
├── Readme.md                     # Project documentation
├── requirements.txt              # Python dependencies
└── tests
    ├── requirements.txt          # Testing dependencies
    ├── test_logger.py            # Unit tests for logger.py
    ├── test_main.py              # Unit tests for main.py
    └── test_utils.py             # Unit tests for utils.py
```

---

## 📑 API Docs

Once running, visit:

- Swagger UI: `http://localhost:8000/docs`
- Redoc: `http://localhost:8000/redoc`

---

## 🤝 Contributing

1. Create a new feature branch
2. Implement your changes
3. Submit a pull request

---

## 🪪 License

This project is licensed under the MIT License.
