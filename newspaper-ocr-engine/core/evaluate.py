import cv2
import numpy as np
import time

try:
    import pytesseract
except ImportError:
    pytesseract = None

class OCREvaluator:
    def __init__(self):
        """
        Loads the available OCR engines for evaluation.
        """
        # We can dynamically load TrOCR or Surya here if available
        pass
        
    def evaluate_crop(self, image: np.ndarray, engine: str) -> dict:
        """
        Run the chosen OCR engine on a specific image crop.
        """
        start_time = time.time()
        text = ""
        
        if engine == "tesseract":
            if pytesseract is None:
                text = "[Error: pytesseract not installed]"
            else:
                # Use --psm 7 for single text line, or --psm 3 for block
                # Since the user might select a block or a line, let's use 6 (Assume a single uniform block of text)
                custom_config = r'--oem 3 --psm 6'
                try:
                    text = pytesseract.image_to_string(image, lang='ben', config=custom_config).strip()
                except Exception as e:
                    text = f"[Tesseract Error: {str(e)}]"
                    
        elif engine == "surya":
            # Placeholder: In a real environment, you'd import Surya here
            text = "[Surya Placeholder: Requires surya-ocr package and weights]"
            
        elif engine == "trocr":
            # Placeholder: In a real environment, you'd import your custom TrOCR here
            text = "[TrOCR Placeholder: Requires custom weights]"
            
        else:
            text = f"[Unknown engine: {engine}]"
            
        time_ms = int((time.time() - start_time) * 1000)
        
        return {
            "engine": engine,
            "extracted_text": text,
            "time_ms": time_ms
        }
