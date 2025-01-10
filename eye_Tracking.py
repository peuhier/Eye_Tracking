import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:/Users/hurli/Documents/Eye_tracking/shape_predictor_68_face_landmarks.dat')
def midpoint(p1,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
while True:

    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)

    for face in faces:
        x1, y1 = face.left(),face.top()
        x2 ,y2 = face.right(), face.bottom()

        landmarks = predictor(frame, face)
        left_eye_region = np.array([
            (landmarks.part(36).x, landmarks.part(36).y),
            (landmarks.part(37).x, landmarks.part(37).y),
            (landmarks.part(38).x, landmarks.part(38).y),
            (landmarks.part(39).x, landmarks.part(39).y),
            (landmarks.part(40).x, landmarks.part(40).y),
            (landmarks.part(41).x, landmarks.part(41).y)
        ], dtype=np.int32)

        min_x = min(left_eye_region[:,0])
        min_y = min(left_eye_region[:,1])
        max_x = max(left_eye_region[:,0])
        max_y = max(left_eye_region[:,1])

        eye_frame = frame[min_y:max_y, min_x:max_x]


        eye_frame = cv2.resize(eye_frame, (0, 0), fx=5, fy=5)
        eye_frame = cv2.GaussianBlur(eye_frame,(7,7),0)
        _,threshold = cv2.threshold(eye_frame,50,255,cv2.THRESH_BINARY_INV)
        contours,_ = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            cv2.drawContours(eye_frame,[cnt],-1,(0,0,255),3)
        cv2.imshow('threshold',threshold)
        cv2.imshow('eye',eye_frame)
    cv2.imshow('frame', frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()