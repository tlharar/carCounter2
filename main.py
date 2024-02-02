import cv2
from ultralytics import YOLO
from sortyolov8 import carCounter
x_start, y_start, x_end, y_end = 0, 0, 0, 0
cropping = False
refPoint = []

cap = cv2.VideoCapture("highway.mp4")
model = YOLO("yolov8n.pt")
ret, frame = cap.read()

def mouse_crop(event, x, y, flags, param):
    
    global x_start, y_start, x_end, y_end, cropping, frame, cropped, refPoint
 
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y

   
    elif event == cv2.EVENT_LBUTTONUP:
        x_end, y_end = x, y
        cropping = False 

        refPoint = [(x_start, y_start), (x_end, y_end)]
    

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", mouse_crop)


def cropFrame():
    while True:
        ret, frame = cap.read()

        
        if len(refPoint)== 2:
            try:
                carCounter(cap, model,y_start,y_end, x_start,x_end)
            except:
                print("Kesilemedi")
                break
            
        
        if cropping:
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("frame", frame)

        if cv2.waitKey(20) & 0xFF == ord("q"):
            break


cropFrame()
cap.release()
cv2.destroyAllWindows()
