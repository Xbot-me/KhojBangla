import cv2
from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor
from PIL import Image
import surya.settings

surya.settings.LANGUAGES = ["bn"]
fp = FoundationPredictor()
predictor = RecognitionPredictor(fp)

img = cv2.imread('downloads/1980_jan1_pg1.jpg')
pil_img = Image.fromarray(img).convert("RGB")
bboxes = [[[100, 100], [400, 100], [400, 150], [100, 150]]] # Surya expects polygon or bbox as [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

res = predictor([pil_img], polygons=[bboxes])
print("Surya output:", res)
