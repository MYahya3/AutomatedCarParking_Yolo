# Import Libraries:
import numpy as np
import torch
import cv2
import easyocr
import datetime
import time
import pandas as pd


def read_text(img):
    #Convert to Grey scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # initialize the easyocr Reader object
    OCR = easyocr.Reader(["en"], gpu=False)
    # detect text
    ocr_result = OCR.readtext(img)
    return ocr_result

# Function to filter number-plate text
def filter_text(region, ocr_result, region_threshold= 0.1):
    rectangle_area = region.shape[0] * region.shape[1]
    text = []
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length * height / rectangle_area > region_threshold:
            text.append(result[1])
    return text

# Convert time string to seconds
def get_sec(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return (int(datetime.timedelta(hours=int(h),minutes=int(m),seconds=int(s)).total_seconds()))

# Calculate fare w.r.t difference between entry and exit time
def fare_calculation(output):
    v = []
    for values in output.Time:
        v.append(get_sec(values))

    g = v[1] - v[0]
    if g < 3600:
        hours = 1
        rate = hours * 1  # Charge 1$/hour
    else:
        hours = int(g / 3600) + 1
        rate = hours * 1

    print("Vehicle Removed!\n" \
          "License_number: {}".format(output.License_number[0]) +"\n" 
          "Your Total for " + "{:.2f}".format(hours) + " hours is $" + "{:.2f}".format(rate))


                                            ### Load Model ###

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', 'E:/GitHub/last.pt', force_reload=True)

# Initialize Video
cap = cv2.VideoCapture("E:/GitHub/data-Videos/number_plate.mp4")

lis = []
count = 0
                                                      ### While Loop ###
while True:
    ret, frame = cap.read()  # Grab image and read (720,1280)
    if ret:
        count += 15  # i.e. at 30 fps, this advances one second
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
    else:
        break

    # roi = frame
    roi = frame[390:596,0:1090]
    # Make detection
    result = model(roi)

    # Extract cords, score, label
    for car in result.xyxy[0]:
        x, y, w, h, conf, label = car

        # For Car
        if int(label) == 0:
            cv2.rectangle(roi, (int(x), int(y)), (int(w), int(h)), (70,150, 80), 2)

        # For number Plate
        if int(label) == 1 and conf > 0.50:  # for
            cv2.cv2.rectangle(roi, (int(x), int(y)), (int(w), int(h)), (0,0, 255), 2)

            # Crop the numberplate
            plateROI = roi[int(y):int(h), int(x):int(w)]
            # Extract License_number
            ocr_result = read_text(plateROI)
            # Filter text
            text = filter_text(plateROI, ocr_result, region_threshold=0.2)

            # Create a dict to save text with time
            dict = {'Time': time.strftime("%T"),'License_number': str(text)[1:-1]}
            lis.append(dict)

            # Save unique and initial detected output
            if len(lis) >= 2:
                if lis[-2]["License_number"] == lis[-1]["License_number"]:
                    lis.pop(-1)
    # Create Dataframe
    df = pd.DataFrame(lis, columns=['Time', 'License_number'], index=None)

    # For exit of car
    if df.duplicated(subset=['License_number']).any():
        duplicate = df[df.duplicated('License_number', keep=False)]
        fare_calculation(duplicate)
    # After exit drop vechile entry/exit info from data
    df = df.drop_duplicates(subset=["License_number"],keep=False).reset_index(drop=True)

    df.to_csv('E:/GitHub/parking_history.csv', index=False )
    print(df)

    # Display the resulting frame
    cv2.imshow("Car_detection", frame[0:720,0:1120])
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# Release the capture
cap.release()
# Destroying All the windows
cv2.destroyAllWindows()