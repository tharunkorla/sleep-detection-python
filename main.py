import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

sleep = 0
active = 0
status = "ACTIVE"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face)

        if len(eyes) == 0:
            sleep += 1
            active = 0
            if sleep > 5:
                status = "SLEEPING"
        else:
            active += 1
            sleep = 0
            if active > 5:
                status = "ACTIVE"

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.putText(frame, status, (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (0,0,255), 3)

    cv2.imshow("Sleep Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
