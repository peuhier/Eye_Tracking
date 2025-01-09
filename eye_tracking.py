import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:/Users/hurli/Documents/Eye_tracking/shape_predictor_68_face_landmarks.dat')

def midpoint(p1,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def get_gaze_ratio(eye_points,facial_landmarks):
    left_eye_region = np.array([
            (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
            (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
            (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
            (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
            (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
            (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)
        ], dtype=np.int32)
        
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = min(left_eye_region[:,0])
    min_y = min(left_eye_region[:,1])
    max_x = max(left_eye_region[:,0])
    max_y = max(left_eye_region[:,1])

    eye_frame = eye[min_y:max_y, min_x:max_x]
    _, threshold = cv2.threshold(eye_frame, 70, 255, cv2.THRESH_BINARY)
    height_eye,width_eye = eye_frame.shape

    left_side_threshold = threshold[0:height_eye,0:int(width_eye/2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold[0:height_eye,int(width_eye/2):width_eye]
    right_side_white = cv2.countNonZero(right_side_threshold)
    if left_side_white ==0:
        return 1
    gaze_ratio = right_side_white/(left_side_white)

    return gaze_ratio

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

        
        gaze_ratio_left_eye = get_gaze_ratio([36,37,38,39,40,41],landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42,43,44,45,46,47],landmarks)
        gaze_ratio = (gaze_ratio_left_eye+gaze_ratio_right_eye)/2

        cv2.putText(frame, str(gaze_ratio), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if gaze_ratio<0.7:
            cv2.putText(frame, "LEFT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif 0.7<gaze_ratio<3:
            cv2.putText(frame, "CENTER", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "RIGHT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
