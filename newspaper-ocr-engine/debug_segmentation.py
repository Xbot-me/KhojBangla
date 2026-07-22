import cv2
import numpy as np
import os
import sys

# Add current dir to path to import core modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.preprocess import preprocess_image
from core.segment import DocumentSegmenter

def run_debug(image_path: str):
    print(f"Loading {image_path}...")
    original = cv2.imread(image_path)
    
    print("Preprocessing...")
    try:
        from skimage.filters import threshold_sauvola
        preprocessed = preprocess_image(image_path)
    except ImportError:
        print("scikit-image not installed in this env, falling back to simple threshold for debug visualization...")
        # fallback for debug if scikit-image is not in venv
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        _, preprocessed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        preprocessed = cv2.bitwise_not(preprocessed) # Make background white
        
    cv2.imwrite("debug_1_preprocessed.jpg", preprocessed)
    
    segmenter = DocumentSegmenter()
    print("Detecting Layout (Columns)...")
    paragraphs = segmenter.detect_paragraphs(preprocessed)
    
    # Draw columns
    debug_cols = original.copy()
    for p in paragraphs:
        x1, y1, x2, y2 = p["bbox"]
        cv2.rectangle(debug_cols, (x1, y1), (x2, y2), (0, 0, 255), 4) # Red for columns
        
    cv2.imwrite("debug_2_columns.jpg", debug_cols)
    
    # Draw lines
    print("Detecting Lines via HPP...")
    debug_lines = debug_cols.copy()
    
    for p in paragraphs:
        x1, y1, x2, y2 = p["bbox"]
        para_crop = preprocessed[y1:y2, x1:x2]
        
        # We need to simulate the HPP to get the bounding boxes instead of just the image crops
        # so we can draw them on the global image.
        if para_crop.mean() > 127:
            binary_inv = cv2.bitwise_not(para_crop)
        else:
            binary_inv = para_crop
            
        hpp = np.sum(binary_inv, axis=1)
        if hpp.max() > 0:
            hpp = (hpp / hpp.max()) * 255
            
        threshold = 5
        is_text = hpp > threshold
        
        in_line = False
        start_y = 0
        
        for y, val in enumerate(is_text):
            if val and not in_line:
                in_line = True
                start_y = y
            elif not val and in_line:
                in_line = False
                end_y = y
                if end_y - start_y > 5:
                    # Draw green box for line
                    cv2.rectangle(debug_lines, (x1, y1 + start_y), (x2, y1 + end_y), (0, 255, 0), 2)
                    
        if in_line:
            end_y = len(is_text)
            if end_y - start_y > 5:
                cv2.rectangle(debug_lines, (x1, y1 + start_y), (x2, y1 + end_y), (0, 255, 0), 2)

    cv2.imwrite("debug_3_lines.jpg", debug_lines)
    print("Done! Check debug_1_preprocessed.jpg, debug_2_columns.jpg, debug_3_lines.jpg")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img = sys.argv[1]
    else:
        img = "../downloads/1980_jan1_pg1.jpg"
    run_debug(img)
