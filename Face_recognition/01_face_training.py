import os
import cv2 as cv
import numpy as np

people=[]

features=[]
labels=[]

haar_cascade=cv.CascadeClassifier('haar_face.xml')

DIR=r'E:\GIT_PROJECTS\Project_1\Face_recognition\training_data'
for i in os.listdir(DIR):
    people.append(i)

print(people)

people=['Cillian Murphy', 'Jackie Chan', 'Jennifer Lawrence', 'Leonardo Di Caprio', 'Sai Pallavi']

def create_train():
    for person in people:
        path=os.path.join(DIR,person)
        label=people.index(person)

        for img in os.listdir(path):
            img_path=os.path.join(path,img)
            img_array=cv.imread(img_path)

            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

            face_rect=haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for x,y,w,h in face_rect:
                face_roi=gray[y:y+h, x:x+h]
                features.append(face_roi)
                labels.append(label)

create_train()

print('Training Done')
print()

print(f'Length of features is {len(features)}')
print(f'Length of Labels is {len(labels)}')

features=np.array(features, dtype='object')
labels=np.array(labels)

print()

face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

face_recognizer.save('trainer.yml')

np.save('features', features)
np.save('labels', labels)




































