import cv2
from deepface import DeepFace

# Load the face detection classifier
face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture
video_cap = cv2.VideoCapture(0)

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
        
        # Use DeepFace to predict gender and age
        try:
            result = DeepFace.analyze(face_roi, actions=['gender', 'age'], enforce_detection=False)
            gender = result[0]['gender']
            age = result[0]['age']

            # Display gender and age on the video feed
            cv2.putText(video_data, f"Gender: {gender}", (x, y + h + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.putText(video_data, f"Age: {age}", (x, y + h + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        except Exception as e:
            print("Error in gender/age detection:", e)

    # Display the resulting frame with face, gender, and age detection
    cv2.imshow("video_live", video_data)

    # Exit the loop when the 'a' key is pressed
    if cv2.waitKey(10) & 0xFF == ord("a"):
        break

# Release the video capture object and close all windows
video_cap.release()
cv2.destroyAllWindows()
