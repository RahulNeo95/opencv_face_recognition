import cv2 as cv

img=cv.imread('test_image.png')
grey=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

haar_cascade=cv.CascadeClassifier('Cascade/haar_face.xml')

face_rect=haar_cascade.detectMultiScale(grey,scaleFactor=1.1,minNeighbors=2)

for x,y,w,h in face_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
    cv.imshow('Faces', img)

cv.waitKey(0)
cv.destroyAllWindows()




