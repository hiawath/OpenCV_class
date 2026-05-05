import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

# 1. Define the Lifespan manager separately
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function handles the startup and shutdown logic.
    It is the modern, recommended way to handle resources in FastAPI.
    """
    # --- STARTUP LOGIC ---
    print("--- Starting up: Initializing Camera ---")
    # Initialize the camera (0 is usually the default webcam)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open video device.")
        yield
        return

    # Store the camera object in the app state so routes can access it
    app.state.camera = camera
    print("--- Camera initialized successfully ---")

    yield  # The application runs while this is "suspended"

    # --- SHUTDOWN LOGIC ---
    print("--- Shutting down: Releasing Camera ---")
    camera.release()
    print("--- Camera released ---")


# 2. Initialize the FastAPI app with the lifespan handler
app = FastAPI(lifespan=lifespan)


# 3. Define the Video Streaming Generator
async def video_frame_generator(camera):
    """
    A generator that reads frames from the camera and yields them
    in the multipart/x-mixed-replace format (MJPEG).
    """
    while True:
        success, frame = camera.read()
        if not success:
            break
    # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Convert to bytes for streaming
        frame_bytes = buffer.tobytes()

    # Yield the frame in the correct multipart format
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

        # Small sleep to prevent overwhelming the CPU
        await asyncio.sleep(0.03) # Roughly 30 FPS


# 4. Define the Streaming Route
@app.get("/video_feed")
async def video_feed(response: StreamingResponse = None):
    """
    This endpoint serves the live video stream.
    """
    # Retrieve the camera from the app state we set in lifespan
    camera = app.state.camera
    return StreamingResponse(
        video_frame_generator(camera),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/")
async def index():
    """
    Simple root endpoint to check if the server is running.
    """
    return {"message": "Server is running. Visit /video_feed to see the stream."}

import cv2

if __name__ == "__main__":
    # Run the server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

