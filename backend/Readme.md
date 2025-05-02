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
.env\Scriptsctivate  # Windows
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

## 🔐 Authentication Guide

GreenEye uses **JWT-based authentication** to secure access to protected endpoints like model prediction.

### ➕ Register a new user

```bash
curl -X POST "http://localhost:8000/register?username=admin&password=123456"
```

### 🔑 Login and get access token

```bash
curl -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=123456"
```

The response:

```json
{
  "access_token": "your.jwt.token",
  "token_type": "bearer"
}
```

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
  -H "Authorization: Bearer your.jwt.token" \
  -F "file=@leaf.jpg"
```

---

## 📁 Project Structure

```
backend/
├── main.py     # FastAPI app with inference logic
├── auth.py                # User registration, login, JWT
├── users.json             # User store (JSON)
├── requirements.txt       # Python dependencies
└── Dockerfile             # Deployment container
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
