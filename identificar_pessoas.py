import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose