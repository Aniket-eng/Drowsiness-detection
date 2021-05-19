import cv2
import numpy as np
import winsound
frequency = 2500
duration = 1000
import mediapipe as mp
import time
from keras.models import load_model



lbl=['Close','Open']
mpFace = mp.solutions.face_detection
mpdraw = mp.solutions.drawing_utils
faceDetect = mpFace.FaceDetection()
cap = cv2.VideoCapture(0)
if not cap. isOpened():
	cap = cv2.VideoCapture(1)
if not cap. isOpened():
	raise IOError("Cannot open webcam")   
counter = 0  
pTime = 0
count =0
rpred=[99]
lpred=[99]

model = load_model('cnncat2.h5')
eyes_roi = np.zeros((34,34,3))

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye_tree_eyeglasses.xml')
leye = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_righteye_2splits.xml')


while True:
    ret, frame = cap.read()
    rec.write(frame)
    
    cTime = time.time()
    gray = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
    
    
    eyes = eye_cascade.detectMultiScale(gray,1.1,4) 
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)
    
    for x2,y2,w2,h2 in eyes:
        roi_gray = gray[y2:y2+h2, x2:x2+w2]
        roi_color = frame[y2:y2+h2, x2:x2+w2]
        cv2.rectangle(frame, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 1)
        eyess = eye_cascade.detectMultiScale(roi_gray)
    # Right Eye Prediction
    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict_classes(r_eye)
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break
    #Left Eye Prediction
    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break  

    
    imgRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceDetect.process(imgRgb)

    if results.detections:
        for id,detection in enumerate(results.detections):


            bboxc = detection.location_data.relative_bounding_box
            ih, iw, ic = frame.shape
            bbox = int(bboxc.xmin * iw), int(bboxc.ymin * ih), \
 				   int(bboxc.width * iw), int(bboxc.height * ih)
            x, y, w, h = bbox
            x1, y1 = x + w, y + h
            cv2.rectangle(frame, bbox, (0,255,0), 1)
            # Angle Brackets on Face Box
 			# Top left x,y
            cv2.line(frame, (x, y), (x +30, y), (0, 255, 255), 4)
            cv2.line(frame, (x, y), (x , y + 30), (0, 255, 255), 4)
 			# Top Right x1,y
            cv2.line(frame, (x1, y), (x1 - 30, y), (0, 255, 255), 4)
            cv2.line(frame, (x1, y), (x1, y + 30), (0, 255, 255), 4)
 			# Bottom Left x,y1
            cv2.line(frame, (x, y1), (x + 30, y1), (0, 255, 255), 4)
            cv2.line(frame, (x, y1), (x , y1 - 30), (0, 255, 255),4)
 			# Bottom Right x1,y1
            cv2.line(frame, (x1, y1), (x1 - 30, y1), (0, 255, 255), 4)
            cv2.line(frame, (x1, y1), (x1 , y1 - 30), (0, 255, 255), 4)

 			        

    font =cv2.FONT_HERSHEY_SIMPLEX    
  
    print(rpred)
    print(lpred)
    #Prediction values 1 = Open  & 0 = Closed
    if(rpred[0]==1 and lpred[0]==1):
        status = "Open Eyes"
        cv2.putText(frame,status,(0,150),font,1,(0,255,0),2)
        x1,y1,w1,h1 = 0,0,100,50
        
        cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
        cv2.putText(frame, 'Active', (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)
        if(counter!=0):
            counter = counter-1
    else:
        counter = counter + 1
        status = "Closed Eyes"
        cv2.putText(frame,status,(0,150),font,1,(0,0,255),2)
        x1,y1,w1,h1 = 0,0,100,50
        cv2.rectangle(frame,(x1, y1), (x1+w1, y1+h1), (0, 0, 255), 2)
    cv2.putText(frame,'Score:'+str(counter),(0,450), font, 1,(0,255,0),1,cv2.LINE_AA)
    if counter > 10:   
            
            x1,y1,w1,h1 = 0,0,175,75
            cv2.rectangle(frame, (200, 70), (270 + w1, 110), (0,0,0), -1)
            cv2.putText(frame,'SLEEP ALERT !!',(x1 +200 , y1 + 100 ), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 2)
            winsound.Beep(frequency, duration)
    # Calculating Frames per second(FPS)       
    pTime = time.time() 
    secs = pTime - cTime
    fps = 1/(secs)
    
    cv2.putText(frame,f'FPS: {int(fps)}', (0,100), cv2.FONT_HERSHEY_PLAIN,2,(0,255,0), 2)     
            
    cv2.imshow('Drowsiness Detection',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
    	break

cap.release()
cv2.destroyAllWindows()
