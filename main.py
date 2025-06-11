from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from yolo_detector import something_to_code
import cv2
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/frame")
def get_frame():
    frame, _ = something_to_code()
    _, buffer = cv2.imencode(".jpg", frame)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

@app.get("/faces")
def get_faces():
    _, face_count = something_to_code()
    return JSONResponse(content={"count": face_count})
