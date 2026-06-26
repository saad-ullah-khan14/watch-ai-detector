import cv2
import tensorflow as tf
import numpy as np

# Load AI model
model = tf.keras.models.load_model("watch_detector.h5")

# Camera start
cap = cv2.VideoCapture(0)

print("Camera started... Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize
    img = cv2.resize(frame, (128, 128))

    # Normalize
    img = img / 255.0

    # Expand dimensions
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)

    confidence = prediction[0][0]

    if confidence > 0.6:
        text = "WATCH DETECTED"
        color = (0, 255, 0)   # Green
    else:
        text = "NO WATCH"
        color = (0, 0, 255)   # Red

    # Show confidence score
    conf_text = f"Confidence: {confidence:.2f}"

    # Show text on frame
    cv2.putText(
        frame,
        text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    # Show confidence
    cv2.putText(
        frame,
        conf_text,
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Show camera
    cv2.imshow("Watch AI Detector", frame)

    # Exit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()