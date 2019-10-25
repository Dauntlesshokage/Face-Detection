import cv2

cascPath1 = "haarcascade_frontalface_default.xml"
cascPath2 = "haarcascade_profileface.xml"
facecascade=cv2.CascadeClassifier(cascPath1)
facecascade2=cv2.CascadeClassifier(cascPath2)
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)

while(True):
    ret, frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces2=facecascade2.detectMultiScale(gray,
            scaleFactor=1.3,
            minNeighbors=5)
    faces=facecascade.detectMultiScale(gray,
            scaleFactor=1.3,
            minNeighbors=5)
    for (x,y,w,h) in faces2:
        col=frame[y:y+h,x:x+w]
        img="myimage.jpg"
        cv2.imwrite(img,col)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    for (x,y,w,h) in faces:
        col=frame[y:y+h,x:x+w]
        img="myimage.jpg"
        cv2.imwrite(img,col)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break;

cap.release()
cv2.destroyAllWindows()





