

# **Face Recognition and Emotion Detection System**

## **Overview**
This project implements a real-time **Face Recognition and Emotion Detection System** using Python, OpenCV, and the `FER` library. It captures live video from a webcam, detects faces, and identifies emotions such as happiness, sadness, anger, and more. The system also uses a buffer to smooth out emotion predictions, ensuring reliable and consistent results.

---

## **Features**
- **Face Detection**: Utilizes OpenCV's Haar Cascade Classifier to detect faces in real-time.
- **Emotion Detection**: Uses the `FER` library to predict emotions from detected faces.
- **Confidence Filtering**: Displays emotions only if confidence exceeds a defined threshold.
- **Buffer Smoothing**: Averages emotion predictions over multiple frames for more accurate results.
- **Real-Time Visualization**: Shows detected faces and emotions directly on the video stream.

---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries**:
  - `OpenCV`: For video capturing and face detection.
  - `FER`: For emotion recognition.
  - `NumPy`: For efficient data manipulation.

---

## **How It Works**
1. **Capture Video**: Starts video feed from the default camera using OpenCV.
2. **Detect Faces**: Identifies faces in the frame using Haar cascades.
3. **Predict Emotions**:
   - Analyzes the face region using the `FER` library.
   - Calculates dominant emotion and confidence score.
4. **Display Results**:
   - Draws bounding boxes around detected faces.
   - Displays the most common emotion with a confidence score above the threshold.
5. **Buffer Smoothing**: Maintains a buffer of emotions to provide stable predictions across frames.

---

## **Code Snippet**
Here’s a quick preview of the core functionality:
```python
from fer import FER
import cv2
import numpy as np

# Initialize Emotion Detector and Haar Cascade
emotion_detector = FER()
face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start Video Capture
video_cap = cv2.VideoCapture(0)
while True:
    ret, frame = video_cap.read()
    if not ret:
        break
    # Processing logic here...
    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(10) & 0xFF == ord("a"):
        break
video_cap.release()
cv2.destroyAllWindows()
```

---

## **Setup and Installation**

### **Prerequisites**
Ensure the following tools and libraries are installed:
- Python (>= 3.7)
- OpenCV
- FER Library
- NumPy

### **Installation Steps**
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Face-Recognition-Emotion-Detection.git
   cd Face-Recognition-Emotion-Detection
   ```
2. Install dependencies:
   ```bash
   pip install opencv-python fer numpy
   ```
3. Run the application:
   ```bash
   python main.py
   ```

---

## **Usage**
1. Run the Python script to start the real-time emotion detection system.
2. Press the **'a'** key to exit the video feed.

---

## **Future Enhancements**
- Add age and gender detection for more detailed analytics.
- Integrate a GUI interface for easier interaction.
- Optimize performance for multiple faces in the same frame.
- Store detected emotions for offline analysis.

---

## **Contributing**
Feel free to contribute to this project by:
- Forking the repository.
- Creating a new branch.
- Submitting a pull request with improvements.
