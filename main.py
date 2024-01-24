from deepface import DeepFace
import cv2
import threading
from matplotlib import pyplot as plt
def face_recognition(frame):
    global face_match
    try:
        if DeepFace.verify(frame,imagePathReference.copy())['verified']:
            face_match=True
        else:
            face_match=False
    except ValueError:
        face_match=False




if __name__=='__main__':
    imagePathReference=cv2.imread('reference/16313.jpg')
    face_match=False
    counter=0

    camera=cv2.VideoCapture(0,cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)


    while True:
        success,frame=camera.read()
        if success:
            if counter%30==0:
                try:
                    threading.Thread(target=face_recognition,args=(frame.copy(),)).start()
                except ValueError:
                    pass
            counter+=1
        if face_match:
            cv2.putText(frame,"MATCH!",(20,450),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imshow('FACE',frame)
        cv2.waitKey(1)
    pass