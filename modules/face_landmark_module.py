import cv2
import mediapipe as mp


class FaceLandmark:
    def __init__(self, frame_, max_num_faces=100):
        self.frame_ = frame_
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_landmark_ = self.mp_face_mesh.FaceMesh(max_num_faces=max_num_faces,
                                                         refine_landmarks=True,
                                                         min_detection_confidence=0.5,
                                                         min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def face_landmark(self):
        rgb_image = cv2.cvtColor(self.frame_, cv2.COLOR_BGR2RGB)

        result_face_landmarks = self.face_landmark_.process(rgb_image)

        if result_face_landmarks.multi_face_landmarks:
            for _, facial_landmarks in enumerate(result_face_landmarks.multi_face_landmarks):
                self.mp_drawing.draw_landmarks(image=self.frame_, landmark_list=facial_landmarks,
                                               connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                                               landmark_drawing_spec=None,
                                               connection_drawing_spec=self.mp_drawing_styles
                                               .get_default_face_mesh_tesselation_style())

        # return self.frame_
