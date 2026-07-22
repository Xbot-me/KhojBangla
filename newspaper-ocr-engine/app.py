from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
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
