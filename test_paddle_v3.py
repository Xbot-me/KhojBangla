from paddleocr import PPStructureV3
import cv2

table_engine = PPStructureV3(lang='bn', show_log=False)
img = cv2.imread('downloads/1980_jan1_pg1.jpg')
result = table_engine(img)

for line in result[:5]:
    print(f"BBox: {line['bbox']}")
    # PPStructureV3 returns bounding boxes and maybe types/res depending on the structure
    print(line)
