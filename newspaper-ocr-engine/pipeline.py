import time
from typing import Dict, Any, List
import uuid

from core.preprocess import preprocess_image
from core.segment import DocumentSegmenter
from core.ocr import TrOCREngine
from core.postprocess import LLMCorrector

class OCRPipeline:
    def __init__(self):
        """
        Initialize the complete OCR pipeline.
        Loads all models into memory once.
        """
        print("Initializing DocumentSegmenter...")
        self.segmenter = DocumentSegmenter()
        
        print("Initializing TrOCREngine...")
        self.ocr_engine = TrOCREngine()
        
        print("Initializing LLMCorrector...")
        self.corrector = LLMCorrector()
        
    def process_document(self, image_path: str, doc_id: str = None) -> Dict[str, Any]:
        """
        Run the full 5-step ML pipeline on a document image.
        Returns a structured JSON-serializable dictionary.
        """
        start_time = time.time()
        if not doc_id:
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
            
        try:
            # Step 1: Preprocessing (Sauvola, Deskew, Median Blur)
            preprocessed_img = preprocess_image(image_path)
            
            # Step 2: Document Layout Analysis (DLA)
            paragraphs = self.segmenter.detect_paragraphs(preprocessed_img)
            
            results = []
            
            for para in paragraphs:
                para_id = para["id"]
                x1, y1, x2, y2 = para["bbox"]
                
                # Crop the paragraph from the preprocessed image
                para_crop = preprocessed_img[y1:y2, x1:x2]
                
                # Step 3: Line Segmentation (HPP)
                line_crops = self.segmenter.slice_lines_hpp(para_crop)
                
                # Step 4: Text Recognition (TrOCR Batched)
                raw_lines = self.ocr_engine.recognize_lines(line_crops, batch_size=16)
                
                # Combine raw lines into paragraph text
                raw_ocr_text = " ".join(raw_lines)
                
                # Step 5: Post-Processing & LLM Correction
                # Send the whole paragraph to the LLM for context-aware correction
                corrected_text = self.corrector.correct_texts([raw_ocr_text])[0]
                
                results.append({
                    "paragraph_id": para_id,
                    "type": para["type"],
                    "bbox": [x1, y1, x2, y2],
                    "raw_ocr": raw_ocr_text,
                    "corrected_text": corrected_text
                })
                
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return {
                "document_id": doc_id,
                "status": "success",
                "processing_time_ms": processing_time_ms,
                "content": results
            }
            
        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            return {
                "document_id": doc_id,
                "status": "error",
                "error_message": str(e),
                "processing_time_ms": processing_time_ms
            }

# Singleton instance to be used by the FastAPI app
# We lazy-load it in app.py to avoid loading weights during test discovery if needed
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = OCRPipeline()
    return _pipeline
