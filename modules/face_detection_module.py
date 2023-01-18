import cv2
import mediapipe as mp


class FaceDetection:
    def __init__(self, frame_):
        self.frame_ = frame_
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection_ = self.mp_face_detection.FaceDetection(0.75)

    def rectangle_frame(self, rectangle_box, length=25, thickness=4, rect_width=1):
        left, top, right, bottom = rectangle_box
        x1, y1 = left + right, top + bottom

        # rectangles are left, top, right, bottom ...
        cv2.rectangle(self.frame_, rectangle_box, (222, 196, 176), rect_width)

        # top left
        cv2.line(self.frame_, (left, top), (left + length, top), (222, 196, 176), thickness)
        cv2.line(self.frame_, (left, top), (left, top + length), (222, 196, 176), thickness)

        # top right
        cv2.line(self.frame_, (x1, top), (x1 - length, top), (222, 196, 176), thickness)
        cv2.line(self.frame_, (x1, top), (x1, top + length), (222, 196, 176), thickness)

        # bottom left
        cv2.line(self.frame_, (left, y1), (left + length, y1), (222, 196, 176), thickness)
        cv2.line(self.frame_, (left, y1), (left, y1 - length), (222, 196, 176), thickness)

        # bottom right
        cv2.line(self.frame_, (x1, y1), (x1 - length, y1), (222, 196, 176), thickness)
        cv2.line(self.frame_, (x1, y1), (x1, y1 - length), (222, 196, 176), thickness)

        return self.frame_

    def face_detection(self, no_of_face=0):

        rgb_image = cv2.cvtColor(self.frame_, cv2.COLOR_RGB2BGR)

        result_face_detection = self.face_detection_.process(rgb_image)

        height, width, _ = self.frame_.shape

        if result_face_detection.detections:
            for _, detection in enumerate(result_face_detection.detections):
                relative_bounding_box = detection.location_data.relative_bounding_box

                bbox = int(relative_bounding_box.xmin * width), int(relative_bounding_box.ymin * height), int(
                    relative_bounding_box.width * width), int(relative_bounding_box.height * height)

                frame = self.rectangle_frame(bbox)
                cv2.putText(frame, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (222, 196, 176), 2)

                no_of_face = no_of_face + 1

        return self.frame_, no_of_face
