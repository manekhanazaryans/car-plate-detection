#  Real-Time Armenian License Plate Detection (YOLOv5 + Streamlit)

A real-time license plate detection project using **YOLOv5** and **Streamlit**.

## 📖 Overview
This project detects and highlights Armenian car license plates in images or live video.  
It was trained on a **custom dataset of 2700+ labeled images**.

The app provides a simple **Streamlit web interface** for uploading an image and viewing detected plates.

---

##  Tech Stack
- Python 3.12
- PyTorch + YOLOv5
- Streamlit
- OpenCV
- Pillow
- NumPy

---

## Features
- Real-time plate detection
- Streamlit-based GUI for image uploads
- Trained on custom dataset
- Simple and fast deployment

---

##  How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/car-plate-detection-yolov5.git
   cd car-plate-detection-yolov5
2. Install dependencies:

   pip install -r requirements.txt


3. Run the Streamlit app:

   streamlit run app.py

##  Folder Structure

car-plate-detection-yolov5/
├── app.py                # Streamlit web app
├── ocr_plate_reader.py   # OCR post-processing
├── realtime_camera.py    # Real-time video detection
├── yolov5/               # YOLOv5 model folder
├── data.yaml             # Dataset configuration
├── convert_annotations.py
└── split_dataset.py

##  Results

Example detection:![alt text](screenshot.png)

## License

MIT License © 2025 Mane Khanazaryan
