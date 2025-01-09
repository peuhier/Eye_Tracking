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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        x1, y1 = face.left(),face.top()
        x2 ,y2 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        left_point_R = (landmarks.part(36).x, landmarks.part(36).y)
        right_point_R = (landmarks.part(39).x, landmarks.part(39).y)
        top_point_R = midpoint(landmarks.part(37),landmarks.part(38))
        bottom_point_R = midpoint(landmarks.part(41),landmarks.part(40))

        left_point_L = (landmarks.part(42).x, landmarks.part(42).y)
        right_point_L = (landmarks.part(45).x, landmarks.part(45).y)
        top_point_L = midpoint(landmarks.part(43),landmarks.part(44))
        bottom_point_L = midpoint(landmarks.part(46),landmarks.part(47))

        hor_line_R = cv2.line(frame, left_point_R,right_point_R,(0,255,0),2)
        ver_line_R = cv2.line(frame, top_point_R,bottom_point_R,(0,255,0),2)
        hor_line_L = cv2.line(frame, left_point_L,right_point_L,(0,255,0),2)
        ver_line_L = cv2.line(frame, top_point_L,bottom_point_L,(0,255,0),2)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
