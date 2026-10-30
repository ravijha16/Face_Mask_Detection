
import cv2
import numpy as np
import tensorflow as tf

# Load saved AI model
model = tf.keras.models.load_model("face_mask_model.keras")

# Load face detection classifier
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if face_detector.empty():
    raise RuntimeError("Face detector could not be loaded!")

print("Face Mask Detection Started!")
print("Press Q to close webcam.")

# Start webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Webcam not found. Check camera connection."
    )

while True:

    success, frame = camera.read()

    if not success:
        print("Unable to read webcam frame.")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    for (x, y, w, h) in faces:

        # Crop face from original color frame
        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            continue

        # Resize image for AI model
        face = cv2.resize(face, (128, 128))

        # Convert BGR to RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Prepare image for prediction
        face = np.expand_dims(face, axis=0)

        # Predict
        prediction = model.predict(face, verbose=0)[0][0]

        # 0 = With Mask
        # 1 = Without Mask
        confidence = float(prediction)

        if confidence >= 0.5:

            label = "WITHOUT MASK"
            percentage = confidence * 100
            color = (0, 0, 255)

        else:

            label = "WITH MASK"
            percentage = (1 - confidence) * 100
            color = (0, 255, 0)

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            color,
            2
        )

        # Display label and confidence
        text = f"{label}: {percentage:.1f}%"

        cv2.putText(
            frame,
            text,
            (x, max(y-10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

    # Display webcam window
    cv2.imshow("AI Face Mask Detection System", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera
camera.release()
cv2.destroyAllWindows()

print("Webcam Detection Stopped!")
