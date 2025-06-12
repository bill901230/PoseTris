import math
import mediapipe as mp
import os, time, math
import cv2

CAP_DIR = "captures"
PROCESSED = os.path.join(CAP_DIR, "done")
os.makedirs(PROCESSED, exist_ok=True)

mp_pose    = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1,
                    min_detection_confidence=0.5)

def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(
        math.atan2(y3 - y2, x3 - x2) 
      - math.atan2(y1 - y2, x1 - x2)
    )
    
    if angle < 0:
        angle += 360

    angle = min(angle, 360 - angle)
    return angle

def dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def near(angle, target, tol=30):
    return abs((angle - target + 180) % 360 - 180) <= tol

def classifyPose(landmarks):
    label = 0
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   

    # Get the angle between the left elbow, shoulder and hip points. 
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # Get the angle between the right hip, shoulder and elbow points. 
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Get the angle between the left hip, knee and ankle points. 
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    left_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])

    # 计算右侧髖部（躯干–髖–腿）的角度
    right_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])

    thresh = 100
    d1 = dist(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    d2 = dist(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
    d3 = dist(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
    d4 = dist(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value], landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
    d5 = dist(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value], landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
    d6 = dist(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])

    # print("=======================================")
    # print("left_elbow_angle:", int(left_elbow_angle))
    # print("right_elbow_angle:", int(right_elbow_angle))
    # print("left_shoulder_angle:", int(left_shoulder_angle))
    # print("right_shoulder_angle:", int(right_shoulder_angle))
    # print("left_knee_angle:", int(left_knee_angle))
    # print("right_knee_angle:", int(right_knee_angle))
    # print("left_hip_angle:", int(left_hip_angle))
    # print("right_hip_angle:", int(right_hip_angle))
    # print("d1:", int(d1))
    # print("d2:", int(d2))
    # print("d3:", int(d3))
    # print("d4:", int(d4))
    # print("d5:", int(d5))
    # print("d6:", int(d6))
    # print("=======================================")

    # I
    if near(left_shoulder_angle, 0) and near(right_shoulder_angle, 0) and near(left_elbow_angle, 180) and near(right_elbow_angle, 180):
        if near(left_knee_angle, 180) and near(right_knee_angle, 180):
            label = 1
    # L
    if near(left_shoulder_angle, 0) and near(right_shoulder_angle, 90) and near(left_elbow_angle, 180) and near(right_elbow_angle, 180):
        if near(left_knee_angle, 180) and near(right_knee_angle, 180):
            label = 4
    if near(left_shoulder_angle, 90) and near(right_shoulder_angle, 0) and near(left_elbow_angle, 180) and near(right_elbow_angle, 180):
        if near(left_knee_angle, 180) and near(right_knee_angle, 180):
            label = 5

    # T
    if near(left_shoulder_angle, 90) and near(right_shoulder_angle, 90) and near(left_elbow_angle, 180) and near(right_elbow_angle, 180):
        if near(left_knee_angle, 180) and near(right_knee_angle, 180):
            label = 6

    # S
    if near(left_shoulder_angle, 90) and near(left_elbow_angle, 180) and near(left_knee_angle, 90):
            label = 3
    if near(right_shoulder_angle, 90) and near(right_elbow_angle, 180) and near(right_knee_angle, 90):
            label = 2

    # O
    if d1 < thresh and d2 < thresh and d3 < thresh and d4 < thresh and d5 < thresh and d6 < thresh:
            label = 7

    return label
