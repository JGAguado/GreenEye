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
```

### Package Descriptions
- **fastapi:** Modern web framework for building APIs with Python
- **uvicorn:** ASGI server implementation for running FastAPI applications
- **httpx:** Modern async HTTP client for making requests to the ML server
- **pillow:** Python Imaging Library for image processing
- **python-multipart:** Required for handling file uploads in FastAPI

## Setup

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

## Running the Application

Start the FastAPI server:
```sh
python3 model_inference.py
```

The server will start on `http://localhost:8000` by default.

## API Documentation

Once the server is running, you can access:
- Interactive API documentation: `http://localhost:8000/docs`
- Alternative API documentation: `http://localhost:8000/redoc`

## Development

For development purposes, you can run the server with auto-reload:
```sh
uvicorn model_inference:app --reload
```

