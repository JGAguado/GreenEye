# GreenEye Backend

This is the backend service for the GreenEye application, built with FastAPI.

## Dependencies

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
```

### Package Descriptions
- **fastapi:** Modern web framework for building APIs with Python
- **uvicorn:** ASGI server implementation for running FastAPI applications
- **httpx:** Modern async HTTP client for making requests to the ML server
- **pillow:** Python Imaging Library for image processing
- **python-multipart:** Required for handling file uploads in FastAPI
- **timm:** Computer vision model library
- **torch & torchvision:** Deep learning framework and vision utilities
- **numpy:** Numerical computing library
- **python-dotenv:** Environment variable management

## Setup

### Local Development

1. Create a virtual environment (recommended):
```sh
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```sh
pip install -r requirements.txt
```

### Docker Setup

1. Build the Docker image:
```sh
docker build -t greeneye-backend .
```

2. Run the container:
```sh
docker run -p 8000:8000 greeneye-backend
```

## Running the Application

### Local Development
Start the FastAPI server:
```sh
python3 model_inference.py
```

For development with auto-reload:
```sh
uvicorn model_inference:app --reload
```

### Docker
The server will start on `http://localhost:8000` by default.

## API Documentation

Once the server is running, you can access:
- Interactive API documentation: `http://localhost:8000/docs`
- Alternative API documentation: `http://localhost:8000/redoc`

## Project Structure

```
backend/
├── model_inference.py    # Main FastAPI application
├── utils.py             # Utility functions
├── models/              # ML model related code
├── requirements.txt     # Python dependencies
└── Dockerfile          # Docker configuration
```

## Contributing

1. Create a new branch for your feature
2. Make your changes
3. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

