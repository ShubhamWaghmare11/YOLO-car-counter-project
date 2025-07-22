import cv2
import cvzone
from ultralytics import YOLO
from sort import *

model = YOLO("../model/yolov8n.pt")

cap = cv2.VideoCapture("cars.mp4")



classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

mask = cv2.imread("mask.png")
mask = cv2.resize(mask, (1280,720))

tracker = Sort(max_age=20,min_hits=3, iou_threshold=0.3)

limits = [400,297,673,297]

if len(mask.shape) == 2 or mask.shape[2] == 1:
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

totalCount = set()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (1280, 720))  

while True:
    success, img = cap.read()
    imgRegion = cv2.min(img,mask)
    if not success:
        break

    results = model(imgRegion, stream=True)

    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = round(float(box.conf[0]),2)               # Confidence
            cls = box.cls[0]                   # Class 
            
            w, h = x2-x1, y2-y1
            

            currentclass = classNames[int(cls)]

            if currentclass in ['car','truck','bus','motorbike'] and conf > 0.3:
                # Draw rectangle and label

                # cvzone.cornerRect(img,(x1,y1,w,h), l=9, rt = 5)
                # cvzone.putTextRect(img,f"{currentclass} {conf}",(max(0,x1),max(35,y1-20)), scale=1,thickness=1, offset=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections,currentArray))

                resultsTracker = tracker.update(detections)
                cv2.line(img, (limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)
                for result in resultsTracker:
                    x1, y1, x2, y2, Id = map(int,result)
                    w, h = x2-x1, y2-y1

                    cvzone.cornerRect(img,(x1,y1,w,h),l=9, rt=2, colorR=(255,0,0))
                    cvzone.putTextRect(img, f"{currentclass} {Id}", (max(0,x1), max(35,y1)),  scale=2,thickness=5, offset=5)

                    cx, cy = x1 + w//2,y1 + h//2
                    cv2.circle(img, (cx, cy), 5, (255,0,255), cv2.FILLED)

                    if limits[0]< cx < limits[2] and limits[1] -20 < cy < limits[1] + 20:
                        totalCount.add(Id)
                    cvzone.putTextRect(img, f"Count: {len(totalCount)}", (50,50),  scale=2,thickness=5, offset=5)


    # Show image
    cv2.imshow("Image", img)
    out.write(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
