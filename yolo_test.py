from ultralytics import YOLO
import cv2

model = YOLO("best-colab.torchscript")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    if not ret:
        continue

    results = model.predict(frame, imgsz=320, stream=True)
    for result in results:
        for box in result.boxes:
            if box.conf > 0.5:
                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                # Draw rectangle
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
                # Label with confidence
                label = f"{round(float(box.conf), 3)}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # draw a center dot for the bounding box and the entire frame
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                cv2.circle(frame, (int(x_center), int(y_center)), 5, (255, 0, 0), -1)
                cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), 5, (255, 0, 0), -1)

    cv2.imshow("YOLO Results", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()