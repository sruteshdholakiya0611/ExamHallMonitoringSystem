import cv2
import dlib
import mediapipe as mp
from scipy.spatial import distance


class LipsDetection:
    def __init__(self, frame_):
        self.frame_ = frame_

        self.LIPS = [60, 62, 64, 66]

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection_ = self.mp_face_detection.FaceDetection(0.75)

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    @staticmethod
    def get_mouth_ratio(mouth_points, face_landmarks):
        left = (face_landmarks.part(mouth_points[0]).x, face_landmarks.part(mouth_points[0]).y)  # L1
        right = (face_landmarks.part(mouth_points[2]).x, face_landmarks.part(mouth_points[2]).y)  # L5
        top = (face_landmarks.part(mouth_points[1]).x, face_landmarks.part(mouth_points[1]).y)  # L3
        bottom = (face_landmarks.part(mouth_points[3]).x, face_landmarks.part(mouth_points[3]).y)  # L7

        dist1 = distance.euclidean(top, bottom)
        dist2 = distance.euclidean(left, right)

        ratio_ = float(dist1 / dist2)

        return ratio_

    def lips_detection(self):

        height, width, _ = self.frame_.shape

        rgb_image = cv2.cvtColor(self.frame_, cv2.COLOR_BGR2RGB)

        gray_image = cv2.cvtColor(self.frame_, cv2.COLOR_BGR2GRAY)

        result_face_detection = self.face_detection_.process(rgb_image)

        dlib_face_detection = self.detector(gray_image)

        if result_face_detection.detections:

            for face in dlib_face_detection:

                landmarks = self.predictor(gray_image, face)

                mouth_ratio = self.get_mouth_ratio(self.LIPS, landmarks)

                condition = (mouth_ratio > 0.1)

                if condition:
                    cv2.putText(self.frame_, '| [ Mouth is Open! ] |', (20, 250), cv2.FONT_HERSHEY_PLAIN, 2,
                                (123, 104, 238), 2)
                    cv2.putText(self.frame_, "| [ ALERT ] |", (18, 450), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 2)
                    cv2.imwrite("output/main.jpg", self.frame_)
