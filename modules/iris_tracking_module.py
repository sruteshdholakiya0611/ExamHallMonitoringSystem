import cv2
import mediapipe as mp
import numpy as np


class IrisTracking:
    def __init__(self, frame_):
        self.frame_ = frame_

        self.RIGHT_IRIS = [474, 475, 476, 477]
        self.LEFT_IRIS = [469, 470, 471, 472]
        self.L_H_LEFT = [33]
        self.L_H_RIGHT = [133]
        self.R_H_LEFT = [362]
        self.R_H_RIGHT = [263]

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_landmark = self.mp_face_mesh.FaceMesh(max_num_faces=100,
                                                        refine_landmarks=True,
                                                        min_detection_confidence=0.5,
                                                        min_tracking_confidence=0.5)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    @staticmethod
    def iris_distance(iris_center, right_point, left_point):
        center_to_right_dist = np.sqrt(np.sum(np.square(iris_center - right_point)))
        total_distance = np.sqrt(np.sum(np.square(right_point - left_point)))
        iris_ratio = (center_to_right_dist / total_distance)
        if 0.39 <= iris_ratio < 0.55:
            iris_position = "screen"
        else:
            iris_position = "| [ Looking away from screen ] |"

        return iris_position, iris_ratio

    def iris_tracking(self):

        rgb_image = cv2.cvtColor(self.frame_, cv2.COLOR_BGR2RGB)

        height, width, _ = self.frame_.shape

        result_face_landmarks = self.face_landmark.process(rgb_image)

        if result_face_landmarks.multi_face_landmarks:
            # for _, facial_landmarks in enumerate(result_face_landmarks.multi_face_landmarks):
            #     for source_index, target_index in self.mp_face_mesh.FACEMESH_IRISES:
            #         source = facial_landmarks.landmark[source_index]
            #         target = facial_landmarks.landmark[target_index]
            #
            #         relative_source = (int(source.x * self.frame_.shape[1]), int(source.y * self.frame_.shape[0]))
            #         relative_target = (int(target.x * self.frame_.shape[1]), int(target.y * self.frame_.shape[0]))
            #
            #         cv2.line(self.frame_, relative_source, relative_target, (237, 149, 100), 1)
            #         cv2.circle(self.frame_, relative_source, 1, (237, 149, 100), -1)
            #         cv2.circle(self.frame_, relative_target, 1, (237, 149, 100), -1)

            mesh_points = np.array(
                [np.multiply([p.x, p.y], [width, height]).astype(int)
                 for p in result_face_landmarks.multi_face_landmarks[0].landmark])

            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[self.LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[self.RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            cv2.circle(self.frame_, center_left, int(l_radius), (237, 149, 100), 1, cv2.LINE_AA)
            cv2.circle(self.frame_, center_right, int(r_radius), (237, 149, 100), 1, cv2.LINE_AA)
            cv2.circle(self.frame_, mesh_points[self.R_H_RIGHT][0], 2, (193, 193, 193), -1, cv2.LINE_AA)
            cv2.circle(self.frame_, mesh_points[self.R_H_LEFT][0], 2, (183, 183, 183), -1, cv2.LINE_AA)
            cv2.circle(self.frame_, mesh_points[self.L_H_RIGHT][0], 2, (183, 183, 183), -1, cv2.LINE_AA)
            cv2.circle(self.frame_, mesh_points[self.L_H_LEFT][0], 2, (193, 193, 193), -1, cv2.LINE_AA)

            iris_pos, ratio = self.iris_distance(center_right, mesh_points[self.R_H_RIGHT],
                                                 mesh_points[self.R_H_LEFT][0])

            condition = ratio < 0.39 or ratio > 0.55

            if condition:
                cv2.putText(self.frame_, '{}'.format(iris_pos), (20, 150),
                            cv2.FONT_HERSHEY_PLAIN, 2, (123, 104, 238), 2)
                cv2.putText(self.frame_, "| [ ALERT ] |", (18, 450), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 2)
                cv2.imwrite("output/main.jpg", self.frame_)
            else:
                cv2.putText(self.frame_, f'Iris tracking | [ {iris_pos} ] ', (20, 150),
                            cv2.FONT_HERSHEY_PLAIN, 2, (198, 113, 113), 2)
                # (192, 158, 125)
                # (250, 206, 135)
                # (198, 113, 113)
