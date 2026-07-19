import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from banglaocr.ocr_engines import PaddleEngine, SuryaEngine

img = cv2.imread("downloads/1980_jan1_pg1.jpg")
if img is None:
    print("Could not read image")
    sys.exit(1)

# crop the image a bit so it runs fast for testing
crop = img[0:1500, 0:1500]

print("Initializing Surya for layout detection...")
surya = SuryaEngine(["bn"])

print("Initializing Paddle for recognition...")
paddle = PaddleEngine("bengali")

print("Detecting layout...")
res = surya._detection_predictor(images=[crop])[0]
from banglaocr.segment import BoundingBox
bboxes = [BoundingBox(x=b[0], y=b[1], w=b[2]-b[0], h=b[3]-b[1]) for b in res.bboxes]

print(f"Found {len(bboxes)} bounding boxes. Running paddle OCR...")
ocr_res = paddle.ocr_page(crop, bboxes)

print("--- OCR RESULTS ---")
for r in ocr_res[:20]:
    print(f"[{r.confidence:.1f}%] {r.text}")
