import cv2 as cv
import numpy as np
import os

face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainer.yml')

haar_cascade=cv.CascadeClassifier('haar_face.xml')

DIR=r'E:\GIT_PROJECTS\Project_1\Face_recognition\validation_data\sai.jpg'

people=['Cillian Murphy', 'Jackie Chan', 'Jennifer Lawrence', 'Leonardo Di Caprio', 'Sai Pallavi']

img=cv.imread(DIR)

gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

face_rect=haar_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)

for x,y,w,h in face_rect:
    face_roi=gray[y:y+h, x:x+h]

    label, confidence=face_recognizer.predict(face_roi)

    print(f'Label={people[label]} with a confidence of {confidence}.')

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)

    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv.imshow('Face_detected', img)
cv.waitKey(0)














