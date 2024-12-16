import cv2
from fer import FER
import numpy as np

# Initialize the emotion detector
emotion_detector = FER()

# Load the face detection classifier
face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture
video_cap = cv2.VideoCapture(0)

# To store detected emotions over multiple frames
emotion_buffer = []
buffer_size = 10  # Number of frames to average emotions

# Confidence threshold for emotions
confidence_threshold = 0.7

while True:
    # Capture frame-by-frame
    ret, video_data = video_cap.read()

    # Check if the frame was captured correctly
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert to grayscale for face detection
    gray_frame = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cap.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Process each face detected
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Extract the region of interest (the face)
        face_roi = video_data[y:y + h, x:x + w]
        
        # Detect emotions on the face ROI
        emotion_predictions = emotion_detector.detect_emotions(face_roi)
        
        # Check if emotions were detected and append the dominant emotion to buffer
        if emotion_predictions:
            dominant_emotion = max(emotion_predictions[0]['emotions'], key=emotion_predictions[0]['emotions'].get)
            emotion_buffer.append(dominant_emotion)

            # Keep buffer size consistent
            if len(emotion_buffer) > buffer_size:
                emotion_buffer.pop(0)

            # Get the most frequent emotion in the buffer (averaging)
            most_common_emotion = max(set(emotion_buffer), key=emotion_buffer.count)

            # Only display emotion if confidence is above threshold
            confidence = emotion_predictions[0]['emotions'][most_common_emotion]
            if confidence > confidence_threshold:
                cv2.putText(video_data, f"Emotion: {most_common_emotion} ({confidence:.2f})", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame with face and emotion detection
    cv2.imshow("video_live", video_data)

    # Exit the loop when the 'a' key is pressed
    if cv2.waitKey(10) & 0xFF == ord("a"):
        break

# Release the video capture object and close all windows
video_cap.release()
cv2.destroyAllWindows()
