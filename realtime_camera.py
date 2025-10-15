import cv2
import torch
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Optional: Set tesseract path manually if needed (uncomment and update this)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='yolov5/runs/train/plate_detector_v2/weights/best.pt', force_reload=True)

# Start webcam (0 = default, change if needed)
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

print("✅ Running YOLOv5 + OCR on iPhone Camera. Press 'x' to exit.")

last_text = ""  # store the last recognized value

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0].numpy()

    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        plate_img = frame[y1:y2, x1:x2]

        # Resize + Preprocess
        plate_img = cv2.resize(plate_img, None, fx=2, fy=2)
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(thresh, config=config).strip()

        # Update only if OCR result is confident (length ≥ 4)
        if len(text) >= 4:
            last_text = text

        # Draw latest valid result
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, last_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)


    # Show the result
    cv2.imshow("📷 Real-Time License Plate Detection + OCR", frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
