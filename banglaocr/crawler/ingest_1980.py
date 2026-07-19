import os
import sys

def download_first_page(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    file_id = "14zoTjR7xCy33ql-Y7pWScR3UmcNcI_qD"
    drive_url = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
    out_path = os.path.join(output_dir, f"1980_jan1_pg1.jpg")
    
    if os.path.exists(out_path):
        print(f"Already exists: {out_path}")
        return
        
    print(f"Downloading 1st page {file_id} to {out_path}...")
    try:
        import gdown
        result = gdown.download(url=drive_url, output=out_path, quiet=False)
        if not result:
            raise RuntimeError("gdown returned None")
        print("Successfully downloaded 1st page!")
    except Exception as e:
        print(f"Direct download failed ({e}), trying thumbnail fallback...")
        import requests
        thumb_url = f"https://drive.google.com/thumbnail?id={file_id}&sz=w5000-h5000"
        r = requests.get(thumb_url, stream=True)
        r.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print("Successfully downloaded 1st page via thumbnail!")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "../downloads"
    download_first_page(out)
