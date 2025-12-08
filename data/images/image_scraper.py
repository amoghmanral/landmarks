from duckduckgo_search import DDGS
import requests
import csv
import os
from PIL import Image
from io import BytesIO
import time


# ============ SETTINGS ============
IMAGES_PER_LANDMARK = 40
START_FROM = 0
# ==================================


STOCK_DOMAINS = [
    'shutterstock', 'istockphoto', 'gettyimages', 'adobestock',
    'dreamstime', 'depositphotos', 'alamy', '123rf', 'bigstock',
    'stockphoto', 'vectorstock', 'pond5', 'canstockphoto',
    'fotolia', 'eyeem', 'westend61', 'masterfile', 'thinkstock',
]


def is_stock_image(url, image_url):
    """Check if image is from a stock photo site"""
    combined = (url + image_url).lower()
    return any(domain in combined for domain in STOCK_DOMAINS)


def is_valid_image(img_data):
    """Check if image data is valid and readable"""
    try:
        img = Image.open(BytesIO(img_data))
        img.verify()
        return True
    except:
        return False


def download_images_for_landmark(idx, name, country, target_count):
    """Download valid images for a landmark until we have target_count"""
    print(f"Downloading: {name}")
    
    safe_name = name.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
    safe_name = safe_name[:50]
    folder = f"{idx:04d}_{safe_name}"
    folder_path = f"landmark_images/{folder}"
    os.makedirs(folder_path, exist_ok=True)
    
    valid_images = 0
    max_attempts = target_count * 3
    
    queries = [
        f"{name} photograph",
        f"{name} {country} photo",
        f'"{name}" photography',
    ]
    
    for query in queries:
        if valid_images >= target_count:
            break
            
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(
                    query, 
                    max_results=max_attempts,
                    type_image="photo",
                ))

            
            for result in results:
                if valid_images >= target_count:
                    break
                
                image_url = result.get('image', '')
                source_url = result.get('url', '')
                
                if is_stock_image(source_url, image_url):
                    print(f"Stock image, skipping")
                    continue
                
                try:
                    response = requests.get(image_url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                    response.raise_for_status()
                    img_data = response.content
                    
                    if is_valid_image(img_data):
                        filename = f"{folder_path}/{valid_images + 1}.jpg"
                        with open(filename, 'wb') as f:
                            f.write(img_data)
                        valid_images += 1
                        print(f"Valid image {valid_images}/{target_count}")
                    else:
                        print(f"Invalid image")
                        
                except Exception as e:
                    print(f"Download failed")
                    continue
                
                time.sleep(0.2)
        
        except Exception as e:
            print(f"Search failed for query: {query}")
            continue
        
        if valid_images < target_count:
            time.sleep(1)
    
    print(f"Completed: {valid_images}/{target_count} valid images\n")
    return valid_images


# Main
print("Loading landmarks...")
with open('wonders_clean.csv', 'r') as f:
    reader = csv.DictReader(f)
    landmarks = list(reader)

print(f"Found {len(landmarks)} landmarks\n")

successful = 0
failed = []

for idx, landmark in enumerate(landmarks, 1):
    if idx < START_FROM:
        continue
    name = landmark['Name']
    country = landmark['Country']
    
    print(f"[{idx}/{len(landmarks)}] Processing: {name}")
    
    try:
        count = download_images_for_landmark(idx, name, country, IMAGES_PER_LANDMARK)
        if count >= IMAGES_PER_LANDMARK:
            successful += 1
        else:
            failed.append(f"{name} (only {count} images)")
    except Exception as e:
        print(f"Error: {e}\n")
        failed.append(f"{name} (error)")
    
    time.sleep(2)

print(f"\n{'='*60}")
print(f"DONE: {successful} complete, {len(failed)} incomplete")
print(f"Images saved to: landmark_images/")