import pytesseract
from PIL import Image
import pandas as pd

img = Image.open('downloads/1980_jan1_pg1.jpg')
data = pytesseract.image_to_data(img, lang='ben', output_type=pytesseract.Output.DATAFRAME)

# Filter out empty text blocks just to see what hierarchy we get
blocks = data[data['level'] == 2]
paras = data[data['level'] == 3]
lines = data[data['level'] == 4]
words = data[data['level'] == 5]

print(f"Detected {len(blocks)} blocks, {len(paras)} paragraphs, {len(lines)} lines, {len(words)} words.")
print(data[data['level'] == 5].head(10)[['block_num', 'par_num', 'line_num', 'word_num', 'conf', 'text']])
