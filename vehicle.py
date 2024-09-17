import cv2
import numpy as np


cap=cv2.VideoCapture('video.mp4')
count_line_position = 550

algo=cv2.bgsegm.createBackgroundSubtractorMOG()

min_height = 80
min_width = 80


def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx= x+x1
    cy = y+y1
    return cx,cy

detect =[]
offset=6
counter1=0

while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(3,3),5)
    
    img_sub = algo.apply(blur)
    dilate=cv2.dilate(img_sub,np.ones((5,5)))
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernal)
    dilatada = cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE,kernal)
    counter,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame,(25,count_line_position),(1200,count_line_position),(255,127,0),3)

    for (i,c) in enumerate(counter):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>=min_width) and (h>=min_height)
        if not validate_counter:
            continue
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame,center,4,(0,255,0),-1)


        for (x,y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position - offset):
                counter1 += 1
            cv2.line(frame,(25,count_line_position),(1200,count_line_position),(0,127,255),3)
            detect.remove((x,y))
            print("vehicle Counter:"+str(counter1))

    cv2.putText(frame,"VEHICLES: " +str(counter1),(35,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)



    cv2.imshow('Video',frame)

    if cv2.waitKey(1) & 0xff ==27 :
        break

cv2.destroyAllWindows()
cap.release()