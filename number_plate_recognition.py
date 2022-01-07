import numpy as np
import torch
import cv2
import easyocr
# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', 'E:/GitHub/last.pt', force_reload=True)
model.classes = [1]
model.conf = 0.8

# Function to filter number-plate text
def filter_text(region, ocr_result, region_threshold):
    rectangle_area = region.shape[0] * region.shape[1]
    plate = []
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length * height / rectangle_area > region_threshold:
            plate.append(result[1])
    return plate

def resize(image, dim= 820):
    frame = cv2.resize(image, (dim,dim))
    return frame

region_threshold = 0.2
cap = cv2.VideoCapture("E:/GitHub/y.mp4")

while True:
    ret, frame = cap.read()

    roi = frame
    # roi = frame[380:600, 0:1080] # this is for number_plate.mp4 video
    result = model(roi)
    # Get status box
    cv2.rectangle(frame, (0, 0), (300, 150), (255,255,255), -1)

    # Display Plate-Number text
    cv2.putText(frame, 'Plate-Number', (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

    # Extract cords, score, label
    for plate in result.xyxy[0]:
        x, y, w, h, conf, label = plate

        # Draw rectangle when label in ROI
        cv2.rectangle(roi, (int(x), int(y)), (int(w), int(h)), (0, 0, 0), 2)
        # Crop the numberplate
        plateROI = roi[int(y):int(h),int(x):int(w)]
        # initialize the easyocr Reader object
        reader = easyocr.Reader(["en"], gpu=False)
        # detect text
        read_text = reader.readtext(plateROI)
    try:
        text = filter_text(plateROI, read_text, region_threshold)
        # pn = '\n'.join(text)
        cv2.putText(frame, text[0],(80, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 120, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, text[1], (120,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 120, 255), 2, cv2.LINE_AA)
    except:
        pass
    cv2.imshow("frame",resize(frame))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break