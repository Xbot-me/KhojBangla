import cv2
from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor
from PIL import Image
import surya.settings

surya.settings.LANGUAGES = ["bn"]

fp = FoundationPredictor()
predictor = RecognitionPredictor(fp)

img = cv2.imread('downloads/1980_jan1_pg1.jpg')
crop = img[100:150, 100:400]
crop_pil = Image.fromarray(crop)

res = predictor([crop_pil], [["bn"]])
print("Surya output:", res)
