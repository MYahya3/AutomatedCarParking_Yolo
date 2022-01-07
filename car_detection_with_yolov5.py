# Import Libraries:
import torch
import cv2
# Load model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [2]

# Initialize video
cap = cv2.VideoCapture("E:/Cars.mp4")
                                        ### While Loop ###
while True:
    ret, frame = cap.read()  # Grab image and read
    if ret == False:
        break
    # Define region of interest
    roi = frame[100:1080,400:1650]
    roi_text = frame[90:1080, 400:1650]
    # Make detection
    result = model(roi)

    # Extract cords, score, label
    for car in result.xyxy[0]:
        x, y, w, h, conf, label = car

        # Draw rectangle when label in ROI
        cv2.rectangle(roi, (int(x), int(y)), (int(w),int(h)), (255, 0, 0), 2)

        cv2.putText(roi, "Car",(int(x),(int(y)-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv2.imshow("Car_detection",roi)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()