import cv2
import numpy as np
from typing import List, Dict, Any
import os

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class DocumentSegmenter:
    def __init__(self, model_path: str = "weights/yolov8_badlad.pt"):
        self.model = None
        if YOLO is not None:
            # We assume YOLOv8 is available and weights exist, otherwise we fallback or mock
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                # If weights don't exist yet, we instantiate a placeholder model or None
                # print(f"Warning: Model weights not found at {model_path}. Please download them.")
                pass
                
    def detect_paragraphs(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Use YOLOv8 to detect paragraph regions.
        Converts polygons to Minimum Bounding Rectangles (MBR).
        Sorts boxes top-to-bottom, left-to-right.
        """
        # If model is not loaded, we fallback to morphological contour detection
        if self.model is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            # Binarize if not already
            if gray.max() > 1:
                # Assuming background is white, text is black. We need text to be white for dilation.
                if gray.mean() > 127:
                    gray = cv2.bitwise_not(gray)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                thresh = (gray * 255).astype(np.uint8)

            # Morphological dilation to connect text into blocks. 
            # We want to connect words horizontally (large width) and lines vertically (small/medium height)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 10))
            dilated = cv2.dilate(thresh, kernel, iterations=2)
            
            # Find contours of these text blocks
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            paragraphs = []
            h_img, w_img = image.shape[:2]
            
            for i, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Filter out obvious noise or watermarks (too small)
                if w > 50 and h > 20:
                    paragraphs.append({
                        "id": i + 1,
                        "bbox": [x, y, x + w, y + h],
                        "type": "text"
                    })
                    
            # Sort paragraphs top-to-bottom, left-to-right
            paragraphs.sort(key=lambda p: (p["bbox"][1] // 100, p["bbox"][0]))
            
            return paragraphs
            
        # Convert grayscale (if passed) to RGB for YOLO
        if len(image.shape) == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = image

        results = self.model(img_rgb)
        
        paragraphs = []
        for i, box in enumerate(results[0].boxes):
            # xyxy format
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            
            # Optionally map cls_id to type like "text", "headline"
            # Assuming class 0 is text
            block_type = "headline" if cls_id == 1 else "text"
            
            paragraphs.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "type": block_type
            })
            
        # Sort paragraphs: top-to-bottom, left-to-right
        # A simple heuristic: round y1 to nearest 50 pixels to group into rows, then sort by y then x
        paragraphs.sort(key=lambda p: (p["bbox"][1] // 50, p["bbox"][0]))
        
        for i, p in enumerate(paragraphs):
            p["id"] = i + 1
            
        return paragraphs

    def slice_lines_hpp(self, paragraph_img: np.ndarray) -> List[np.ndarray]:
        """
        Use Horizontal Projection Profile (HPP) to slice a paragraph into text lines.
        """
        # Invert image if it's white background (we need text to be white pixels for sum)
        # We assume input is already binarized (white background = 255, black text = 0)
        # So we invert it to count text pixels.
        if paragraph_img.mean() > 127:
            binary_inv = cv2.bitwise_not(paragraph_img)
        else:
            binary_inv = paragraph_img
            
        # Sum along the X-axis
        hpp = np.sum(binary_inv, axis=1)
        
        # Normalize between 0 and 255
        if hpp.max() > 0:
            hpp = (hpp / hpp.max()) * 255
            
        # Threshold to find valleys
        # If HPP is below a certain threshold, we consider it whitespace (valley)
        threshold = 5
        is_text = hpp > threshold
        
        lines = []
        in_line = False
        start_y = 0
        
        for y, val in enumerate(is_text):
            if val and not in_line:
                in_line = True
                start_y = y
            elif not val and in_line:
                in_line = False
                end_y = y
                # Ignore very small noisy lines
                if end_y - start_y > 5:
                    line_crop = paragraph_img[start_y:end_y, :]
                    lines.append(line_crop)
                    
        # Handle case where text touches the bottom of the image
        if in_line:
            end_y = len(is_text)
            if end_y - start_y > 5:
                line_crop = paragraph_img[start_y:end_y, :]
                lines.append(line_crop)
                
        # To do: Edge Case Handling (touching lines based on average line height)
        # Currently a naive split. A robust version would check if (end_y - start_y) > 1.5 * avg_height
        # and forcefully split it in the middle.
        
        return lines
