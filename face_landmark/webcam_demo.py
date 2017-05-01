import sys
import os
import dlib
import glob
from skimage import io
import numpy as np
import cv2

def poseEstimate(size, landmarks):
    image_points = np.array([
            (landmarks[33, 0], landmarks[33, 1]),     # Nose tip
            (landmarks[8, 0], landmarks[8, 1]),       # Chin
            (landmarks[36, 0], landmarks[36, 1]),     # Left eye left corner
            (landmarks[45, 0], landmarks[45, 1]),     # Right eye right corner
            (landmarks[48, 0], landmarks[48, 1]),     # Left Mouth corner
            (landmarks[54, 0], landmarks[54, 1])      # Right mouth corner
        ], dtype="double")

    model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corne
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype = "double")

    dist_coeffs = np.zeros((4,1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    return (p1, p2)


cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280, 720))

predictor_path = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    dets = detector(frame, 0)	
    for k, d in enumerate(dets):
        shape = predictor(frame, d)
        landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
        p1, p2 = poseEstimate(frame.shape, landmarks)
        cv2.line(frame, p1, p2, (255, 0, 0))
        for num in range(shape.num_parts):
            cv2.circle(frame, (shape.parts()[num].x, shape.parts()[num].y), 3, (0,255,0), -1)	
    cv2.imshow('frame', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("q pressed")
        break


cap.release()
out.release()

cv2.destroyAllWindows()



