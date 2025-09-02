import os
import requests
#import ddgs
from ddgs import DDGS

def download_images(query, folder, max_results=20):
    os.makedirs(folder, exist_ok=True)
    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=max_results)
        for i, r in enumerate(results):
            try:
                url = r["image"]
                img_data = requests.get(url, timeout=10).content
                with open(os.path.join(folder, f"{query}_{i}.jpg"), "wb") as f:
                    f.write(img_data)
                print(f"Downloaded {url}")
            except Exception as e:
                print("Error:", e)

if __name__ == "__main__":
    #download_images("cat", "dataset/cats", 50)
    download_images("dog", "dataset/dogs", 50)
