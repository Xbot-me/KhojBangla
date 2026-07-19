import os
import requests
import cv2
import base64
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("VISION_API_KEY")

img = cv2.imread("downloads/1980_jan1_pg1.jpg")
# crop to a small region
crop = img[0:1500, 0:1500]

# encode to jpg
_, buffer = cv2.imencode('.jpg', crop)
content = base64.b64encode(buffer).decode('utf-8')

url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"

payload = {
    "requests": [
        {
            "image": {
                "content": content
            },
            "features": [
                {
                    "type": "TEXT_DETECTION",
                    "model": "builtin/latest"
                }
            ],
            "imageContext": {
                "languageHints": ["bn"]
            }
        }
    ]
}

response = requests.post(url, json=payload)
if response.status_code == 200:
    res = response.json()
    texts = res.get('responses', [{}])[0].get('textAnnotations', [])
    if texts:
        print("Success! Found text:")
        print(texts[0]['description'][:200])
    else:
        print("No text found.")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
