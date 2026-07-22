import requests
import json
import sys

def test_api(image_path: str):
    url = "http://localhost:8000/process"
    
    print(f"Testing API with {image_path}...")
    
    with open(image_path, "rb") as f:
        files = {"file": (image_path.split("/")[-1], f, "image/jpeg")}
        data = {"document_id": "test_scan_1980"}
        
        try:
            response = requests.post(url, files=files, data=data)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("Response JSON:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(f"Error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("Failed to connect to the API. Is uvicorn running on port 8000?")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        image = "../downloads/1980_jan1_pg1.jpg"
    else:
        image = sys.argv[1]
        
    test_api(image)
