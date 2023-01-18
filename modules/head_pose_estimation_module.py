import cv2
import mediapipe as mp
import numpy as np


class HeadPoseEstimation:
    def __init__(self, frame_):
        self.frame_ = frame_
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_landmark = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                                        min_tracking_confidence=0.5,
                                                        refine_landmarks=True, max_num_faces=1)

    def head_pose_estimation(self):

        rgb_image = cv2.cvtColor(self.frame_, cv2.COLOR_RGB2BGR)

        result_face_landmarks = self.face_landmark.process(rgb_image)

        face_3d = []
        face_2d = []

        height, width, _ = self.frame_.shape

        if result_face_landmarks.multi_face_landmarks:
            for facial_landmarks in result_face_landmarks.multi_face_landmarks:

                # for source_index, target_index in self.mp_face_mesh.FACEMESH_IRISES:
                #     source = facial_landmarks.landmark[source_index]
                #     target = facial_landmarks.landmark[target_index]
                #
                #     relative_source = (int(source.x * self.frame_.shape[1]), int(source.y * self.frame_.shape[0]))
                #     relative_target = (int(target.x * self.frame_.shape[1]), int(target.y * self.frame_.shape[0]))
                #
                #     cv2.line(self.frame_, relative_source, relative_target, (237, 149, 100), 1)
                #     cv2.circle(self.frame_, relative_source, 1, (237, 149, 100), -1)
                #     cv2.circle(self.frame_, relative_target, 1, (237, 149, 100), -1)

                for idx, landmarks in enumerate(facial_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:

                        # x, y = int(landmarks.x * width), int(landmarks.y * height)

                        x, y, z = int(landmarks.x * width), int(landmarks.y * height), landmarks.z

                        face_2d.append([x, y])

                        face_3d.append([x, y, z])

                face_2d = np.array(face_2d, dtype=np.float64)

                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * width

                cam_matrix = np.array([[focal_length, 0, height / 2],
                                       [0, focal_length, width / 2],
                                       [0, 0, 1]])

                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                _, rotation_vector, translation_vector = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                rotation_mat, _ = cv2.Rodrigues(rotation_vector)

                angles, _, _, qx, qy, qz = cv2.RQDecomp3x3(rotation_mat)

                x = angles[0] * 360
                y = angles[1] * 360

                if y < -10:
                    text = "Looking | [ Left ]"
                elif y > 10:
                    text = "Looking | [ Right ]"
                elif x > 10:
                    text = "Looking | [ Up ]"
                elif x < -10:
                    text = "Looking | [ Down ]"
                else:
                    text = "Looking | [ Screen ]"

                condition = y < -10 or y > 10 or x > 10 or x < -10

                if condition:
                    cv2.putText(self.frame_, text, (20, 200), cv2.FONT_HERSHEY_PLAIN, 2,
                                (123, 104, 238), 2)
                    cv2.putText(self.frame_, "| [ ALERT ] |", (18, 450), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 2)
                    cv2.imwrite(
                        "output/head-pose"
                        "-estimation-alert.jpg", self.frame_)
                else:
                    cv2.putText(self.frame_, text, (20, 200), cv2.FONT_HERSHEY_PLAIN, 2, (238, 104, 123), 2)
