import cv2 as cv

capture=cv.VideoCapture('test_video.mp4')

haar_cascade=cv.CascadeClassifier('Cascade/haar_face.xml')


while True:
    isTrue, frame=capture.read()
    grey=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    face_rect=haar_cascade.detectMultiScale(grey,scaleFactor=1.1,minNeighbors=2)

    for x,y,w,h in face_rect:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
        cv.imshow('Player', frame)

    if cv.waitKey(1) and 0xFF==ord('d'):
        break

cv.destroyAllWindows()




