from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import cv2
import numpy as np
import uvicorn
import shutil
import os
import uuid

from pipeline import get_pipeline

app = FastAPI(
    title="Historical Bangla OCR API",
    description="Microservice for extracting and correcting historical Bengali document scans.",
    version="1.0.0"
)

# Mount static files
import os
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

from core.evaluate import OCREvaluator
evaluator = OCREvaluator()

class CropRequest(BaseModel):
    image_filename: str
    x: int
    y: int
    w: int
    h: int
    engine: str

@app.post("/api/evaluate-crop")
async def evaluate_crop(req: CropRequest):
    """
    Evaluates a specific cropped region of an image using a selected OCR engine.
    """
    # Construct full path to image in static directory
    img_path = os.path.join(static_dir, "images", req.image_filename)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found")
        
    # Load image
    image = cv2.imread(img_path)
    if image is None:
        raise HTTPException(status_code=500, detail="Failed to load image")
        
    # Validate crop bounds
    img_h, img_w = image.shape[:2]
    x, y, w, h = req.x, req.y, req.w, req.h
    
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        raise HTTPException(status_code=400, detail="Crop coordinates are out of image bounds")
        
    # Crop the image
    crop = image[y:y+h, x:x+w]
    
    # Run evaluation
    result = evaluator.evaluate_crop(crop, req.engine)
    return JSONResponse(content=result)

# Temporary directory for uploaded files
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/process")
async def process_document(
    file: UploadFile = File(...),
    document_id: str = Form(None)
):
    """
    Process a historical newspaper scan through the OCR pipeline.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
        
    if not document_id:
        document_id = f"doc_{uuid.uuid4().hex[:8]}"
        
    # Save the uploaded file temporarily
    temp_file_path = os.path.join(UPLOAD_DIR, f"{document_id}_{file.filename}")
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Get the pipeline singleton
        pipeline = get_pipeline()
        
        # Run inference
        result = pipeline.process_document(temp_file_path, doc_id=document_id)
        
        # Determine HTTP status code based on pipeline success
        if result.get("status") == "error":
            # For demonstration, returning 500 for inference errors
            return JSONResponse(status_code=500, content=result)
            
        return JSONResponse(status_code=200, content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
