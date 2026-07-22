import cv2
import numpy as np
from skimage.filters import threshold_sauvola

def deskew_image(image: np.ndarray) -> np.ndarray:
    """Deskew image using Hough Transform."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # Use Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # Detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    if lines is not None:
        angles = []
        for line in lines:
            rho, theta = line[0]
            # Convert theta to degrees
            angle = (theta * 180 / np.pi) - 90
            # Only consider small skews (e.g., between -15 and 15 degrees)
            if -15 <= angle <= 15:
                angles.append(angle)
        
        if angles:
            # Get the median angle
            median_angle = np.median(angles)
            if abs(median_angle) > 0.1:
                # Rotate image to deskew
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                # Use white background for rotation filling if original was light, else black.
                # Assuming document images have light background.
                rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                return rotated
    return image


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Apply preprocessing to historical newspaper scans.
    1. Deskew
    2. Grayscale & Sauvola Thresholding
    3. Noise Removal (Median Blur)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    # 1. Deskew
    deskewed = deskew_image(image)
    
    # 2. Grayscale & Sauvola Thresholding
    gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
    
    # Sauvola parameters: window_size should be odd and large enough to capture text context
    window_size = 25
    thresh_sauvola = threshold_sauvola(gray, window_size=window_size)
    
    # Binarize: pixels > threshold become white (255), else black (0)
    binary = (gray > thresh_sauvola).astype(np.uint8) * 255
    
    # 3. Noise Removal (Median Blur)
    # Use a lightweight median blur to remove salt-and-pepper noise
    denoised = cv2.medianBlur(binary, 3)
    
    return denoised
