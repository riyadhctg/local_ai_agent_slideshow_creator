import os
import re
import random
from time import sleep
from io import BytesIO
from pathlib import Path
from typing import List
import requests
from PIL import Image
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import RatelimitException


class ImageFetcher:
    def __init__(self, max_images: int = 3, fallback_size: tuple = (1280, 720)):
        self.max_images = max_images
        self.fallback_size = fallback_size

    def search_images(self, query: str, num_results: int = 3) -> List[str]:
        print(f"Searching images for query: {query}")
        return search_images(query, max_images=num_results)

    def extract_urls(self, text: str) -> List[str]:
        return re.findall(r"https?://[^\s,]+", text)

    def download_image(self, url: str, filename: str) -> bool:
        try:
            print(f"Downloading image from: {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            img.verify()
            img = Image.open(BytesIO(response.content))
            img.save(filename)
            print(f"Saved image to: {filename}")
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False

    def get_images_for_segment(self, segment: str, output_dir: str) -> str:
        keywords = re.sub(r"[^\w\s]", "", segment.split(".")[0][:60])
        print(f"Extracted keywords: '{keywords}'")
        image_urls = self.search_images(keywords, num_results=self.max_images)
        os.makedirs(output_dir, exist_ok=True)

        existing_files = sorted(Path(output_dir).glob("img_*.jpg"))
        next_idx = (
            max(
                [
                    int(f.stem.split("_")[-1])
                    for f in existing_files
                    if f.stem.split("_")[-1].isdigit()
                ],
                default=-1,
            )
            + 1
        )
        img_path = os.path.join(output_dir, f"img_{next_idx}.jpg")

        if image_urls:
            for url in image_urls:
                if self.download_image(url, img_path):
                    break
            else:
                self._fallback_image(img_path)
        else:
            print("No image URLs found. Using fallback image.")
            self._fallback_image(img_path)

        sleep(random.uniform(2, 5))
        return img_path

    def _fallback_image(self, path: str):
        width, height = self.fallback_size
        img = Image.new("RGB", (width, height))
        pixels = img.load()
        for y in range(height):
            r = int(135 + (75 * y / height))
            g = int(206 - (106 * y / height))
            b = int(250 - (80 * y / height))
            for x in range(width):
                pixels[x, y] = (r, g, b)
        img.save(path)


def search_images(
    query: str, max_images: int = 5, max_wait: int = 180, max_retries: int = 3
) -> List[str]:
    with DDGS() as ddgs:
        for attempt in range(max_retries):
            try:
                results = ddgs.images(query, max_results=max_images)
                return [r["image"] for r in results if "image" in r]
            except RatelimitException:
                wait_time = min(2**attempt * max_retries, max_wait)
                print(f"Rate limit hit. Retrying in {wait_time} seconds...")
                sleep(wait_time)
        print("Failed to get images after retries. Returning empty list.")
        return []
