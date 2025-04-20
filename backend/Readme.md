Here's a minimal `requirements.txt` file containing all necessary dependencies to run your FastAPI application (`app.py`):

```
fastapi==0.110.0
uvicorn[standard]==0.29.0
httpx==0.27.0
pillow==10.3.0
python-multipart==0.0.9
```

### Explanation:
- **fastapi:** Web framework to build API.
- **uvicorn:** ASGI server to run FastAPI applications.
- **httpx:** For making asynchronous HTTP requests to the ML server.
- **pillow:** For image processing (resizing, reading images).
- **python-multipart:** Required for handling file uploads with FastAPI.

### How to use:
Install these dependencies with the command:
```sh
pip install -r requirements.txt
```

Run your FastAPI app with:
```sh
uvicorn app:app --reload
```

Let me know if you need further assistance!