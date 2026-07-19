from PIL import Image
from surya.layout import LayoutPredictor
from surya.detection import DetectionPredictor

from surya.foundation import FoundationPredictor

image = Image.open("downloads/48.jpg").convert("RGB")
det_predictor = DetectionPredictor()
foundation_predictor = FoundationPredictor()
layout_predictor = LayoutPredictor(foundation_predictor)

layout_results = layout_predictor([image])[0]

for poly in layout_results.polygons:
    print(f"Label: {poly.label}, bbox: {poly.polygon}")
