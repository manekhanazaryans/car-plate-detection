import cv2
import easyocr
from pathlib import Path

reader = easyocr.Reader(['en'])  # Add more langs if needed
image_dir = Path('yolov5/runs/detect/detect_results5')

# Check if directory is correct
print(f"Looking for images in: {image_dir.resolve()}")

# Loop through each .jpg image in detection output
for img_path in image_dir.glob('*.jpg'):
    print(f"\n🔍 Processing: {img_path.name}")
    
    image = cv2.imread(str(img_path))
    
    if image is None:
        print("⚠️ Could not read image.")
        continue

    # Run OCR
    results = reader.readtext(image)

    if not results:
        print("🚫 No text detected.")
    else:
        for (bbox, text, confidence) in results:
            print(f"🟦 Text: {text}, 🔢 Confidence: {confidence:.2f}")
