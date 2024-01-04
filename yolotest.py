import cv2
from ultralytics import YOLO

modelo = YOLO('yolov8n.pt')

img = cv2.imread('road.jpg')

resultado = modelo(img, stream=True)

classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

if resultado:
    for objetos in resultado:
        obj = objetos.boxes
        for dados in obj:
            x, y, w, h = dados.xyxy[0]
            x, y, w, h = int(x), int(y), int(w), int(h)
            conf = int(dados.conf[0]*100)/100
            cls = int(dados.cls[0])
            classe = classes[cls]
            cv2.rectangle(img,(int(x),int(y)),(int(w),int(h)),(255,0,0),3)
            cv2.putText(img,f'{classe} - {round(conf,2)}',(x,y+20),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,0,255),2)


cv2.imshow('Final',img)
cv2.waitKey(0)