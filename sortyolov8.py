import cv2
from sort import *
import math
import numpy as np
from ultralytics import YOLO
import cvzone

cap = cv2.VideoCapture("videostab.mp4")
model = YOLO("yolov8n.pt")

def carCounter(cap, model,y_start,y_end, x_start,x_end):
    counter = 0
    line = [90,300,600,300]
    tracker = Sort(max_age = 20)
    counter = []
    minus = (y_end - y_start)/36
    
    while True:
        ret, frame = cap.read()
        liney = int(y_end - y_start)
        linex = int(x_end - x_start)
        midy = int(liney/2)
        midx = int(linex/2)
        cropped = frame[y_start:y_end, x_start:x_end]
        results = model(cropped, stream=1, classes =[2,3])
        
        detections = np.empty((0,4))
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                newDetections = np.array([x1,y1,x2,y2])
                detections = np.vstack((detections,newDetections))

        trackResults = tracker.update(detections)
        #cv2.line(frame,(line[0],line[1]),(line[2],line[3]), (0,255,255), 6)
        cv2.line(cropped,(0,midy),(x_end,midy),(255,0,0),6)
        for track in trackResults:
            x1,y1,x2,y2,id = track 
            x1,y1,x2,y2,id = int(x1),int(y1),int(x2),int(y2),int(id)
            #cv2.rectangle(frame,(x1,y1),(x2,y2), (0,255,0),1)
            cv2.rectangle(cropped,(x1,y1),(x2,y2), (0,255,0),1)
            #cv2.putText(frame, str(id),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0), 2,cv2.LINE_AA)
            #cv2.putText(cropped,str(id),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255), 2,cv2.LINE_AA)
            w = x2-x1
            h = y2-y1
            cx,cy = x1+w//2, y1+h//2
            #cv2.circle(frame, (cx,cy), 5 , (255,0,0),-1)
            cv2.circle(cropped, (cx,cy), 5 , (255,0,0),-1)
            if 0 < cx < x_end and midy-20 < cy < midy+20 :
                #cv2.line(frame,(line[0],line[1]),(line[2],line[3]), (255,0,0), 6)
                cv2.line(cropped,(0,midy),(x_end,midy),(0,0,255),6)
                if counter.count(id) == 0:
                    counter.append(id)
                
            #cv2.putText(frame, str(len(counter)),(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0), 2,cv2.LINE_AA)
        cv2.putText(cropped,  "Total Counter : " + str(len(counter)),(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0), 2,cv2.LINE_AA)
        cv2.imshow("frame", frame)
        cv2.imshow("cropped", cropped)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break


    cap.release()
    cv2.destroyAllWindows()
