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

        #hor_line_R = cv2.line(frame, left_point_R,right_point_R,(0,255,0),2)
        #ver_line_R = cv2.line(frame, top_point_R,bottom_point_R,(0,255,0),2)
        #hor_line_L = cv2.line(frame, left_point_L,right_point_L,(0,255,0),2)
        #ver_line_L = cv2.line(frame, top_point_L,bottom_point_L,(0,255,0),2)
        
        left_eye_region = np.array([
            (landmarks.part(36).x, landmarks.part(36).y),
            (landmarks.part(37).x, landmarks.part(37).y),
            (landmarks.part(38).x, landmarks.part(38).y),
            (landmarks.part(39).x, landmarks.part(39).y),
            (landmarks.part(40).x, landmarks.part(40).y),
            (landmarks.part(41).x, landmarks.part(41).y)
        ], dtype=np.int32)

        right_eye_region = np.array([
            (landmarks.part(42).x, landmarks.part(42).y),
            (landmarks.part(43).x, landmarks.part(43).y),
            (landmarks.part(44).x, landmarks.part(44).y),
            (landmarks.part(45).x, landmarks.part(45).y),
            (landmarks.part(46).x, landmarks.part(46).y),
            (landmarks.part(47).x, landmarks.part(47).y)
        ], dtype=np.int32)
        
        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)

        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        #cv2.polylines(mask, [right_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)

        left_eye = cv2.bitwise_and(gray, gray, mask=mask)

        min_x = min(left_eye_region[:,0])
        min_y = min(left_eye_region[:,1])
        max_x = max(left_eye_region[:,0])
        max_y = max(left_eye_region[:,1])

        eye_frame = left_eye[min_y:max_y, min_x:max_x]

        _, threshold = cv2.threshold(eye_frame, 70, 255, cv2.THRESH_BINARY)
        eye_frame = cv2.resize(eye_frame,None,fx = 10,fy=10)
        cv2.imshow('eye_frame', eye_frame)
        threshold = cv2.resize(threshold,None,fx = 10,fy=10)
        cv2.imshow('threshold', threshold)
        cv2.imshow('left_eye', left_eye)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
