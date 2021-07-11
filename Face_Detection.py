import numpy as np
import cv2
import os

faceClassifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

name = input()

def CreateDataSet():
    cam = cv2.VideoCapture(0)
    sam=0
    while True:
        Con, img=cam.read()
        if Con:
           cv2.imshow("MY Cam", img)
           gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
           faces=faceClassifier.detectMultiScale(gray_image, 1.3, 5)
           face=np.array([])
           for x,y,w,h in faces:
                #gray_image=cv2.rectangle(gray_image, (x, y), (x+w,y+h), (255,255,255), 3)
                face = gray_image[y:y+h,x:x+w]
                face = cv2.resize(face, (200, 200))
                cv2.imwrite("Images/{}_{}.jpg".format(name, sam), face)
                sam = sam+1
                gray_image = cv2.putText(gray_image, str(sam), (50,50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,0,0), 2)
                cv2.imshow("Gray Cam", gray_image)
           if sam==400:
                print("We collected your faces......")
                break
                
                cv2.imshow("MY FACE", face)
           if cv2.waitKey(1) ==13:
                break
        

    cam.release()

##create data
CreateDataSet()



### tran your model

path = "Images/"
allimg=os.listdir(path)[:-1]
Training_Data=[]
Labels = np.arange(1, len(allimg)+1)
Labels = np.asarray(Labels, dtype=np.int32)

for i in allimg:
    img_path = path +i
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    Narray = np.asarray(image, dtype=np.uint8)
    Training_Data.append(Narray)


#Face_Model = cv2.face_LBPHFaceRecognizer.create()
Face_Model = cv2.face_LBPHFaceRecognizer.create()
Face_Model.train(Training_Data, Labels)
print("Model Trained......")



#play

cam = cv2.VideoCapture(0)
sam=0
while True:
    Con, img=cam.read()
    if Con:
        gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces= faceClassifier.detectMultiScale(gray_image, 1.3, 6)
        for x,y,w,h in faces:
            face = gray_image[y:y+h,x:x+w]
            face = cv2.resize(face, (200, 200))
            
            pred = Face_Model.predict(face)
            if pred[1]<42:
                img = cv2.putText(img, "Hey! anshuman", (50,50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,0,0), 2)
                cv2.rectangle(img, (x, y), (x+w,y+h), (0,255,0), 3)
                #subprocess.call(["Hye! anshuman how can i help you"])
            else:
                img = cv2.putText(img, "Unknown face @", (50,50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,0,0), 2)
                cv2.rectangle(img, (x, y), (x+w,y+h), (0,0,255), 3)
            
        cv2.imshow("MY FACE", img)
        if cv2.waitKey(1) ==13:
            break
        
cam.release()
