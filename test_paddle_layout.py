import cv2
import warnings
from paddleocr import PPStructure, save_structure_res

warnings.filterwarnings("ignore")

# Initialize PPStructure for layout and OCR
# Note: English is the default structure language, let's see if we can use it on Bangla
table_engine = PPStructure(lang='bn', show_log=True)

img_path = 'downloads/1980_jan1_pg1.jpg'
img = cv2.imread(img_path)
result = table_engine(img)

# Print first few layout blocks
for line in result[:5]:
    print(f"Type: {line['type']}, BBox: {line['bbox']}")
    if 'res' in line and line['res']:
        if isinstance(line['res'], list) and len(line['res']) > 0:
            print(f"  First text: {line['res'][0]['text']}")
        elif isinstance(line['res'], dict) and 'html' in line['res']:
            print("  Contains HTML table")

print(f"Total blocks detected: {len(result)}")
