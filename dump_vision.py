import os
import requests
import cv2
import base64
import json
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("VISION_API_KEY")

img = cv2.imread("downloads/1980_jan1_pg1.jpg")

# encode to jpg
_, buffer = cv2.imencode('.jpg', img)
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
                    "type": "DOCUMENT_TEXT_DETECTION"
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
    with open("vision_response_dump.json", "w") as f:
        json.dump(res, f, indent=2)
    print("Dumped response to vision_response_dump.json")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
