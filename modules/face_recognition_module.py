import math
import os

import cv2
import face_recognition
import numpy as np


class FaceRecognition:
    def __init__(self, frame_):
        self.frame_ = frame_

        self.directory_name = 'dataset'
        self.directory_path = os.listdir(self.directory_name)

    def classify_Images(self):
        dataset_images_ = []
        images_name_ = []

        for img in self.directory_path:
            current_image = cv2.imread(f'{self.directory_name}/{img}')
            dataset_images_.append(current_image)
            images_name_.append(os.path.splitext(img)[0].capitalize())

        return dataset_images_, images_name_

    def encodings_Images(self):

        dataset_images_, _ = self.classify_Images()

        encode_list = []

        for img in dataset_images_:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode_image = face_recognition.face_encodings(img)[0]
            encode_list.append(encode_image)

        return encode_list

    @staticmethod
    def face_confidence(face_distance, face_match_threshold=0.6):
        range_ = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range_ * 2.0)

        if face_distance > face_match_threshold:
            return str(round(linear_val * 100, 2)) + '%'
        else:
            value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
            return str(round(value, 2)) + '%'''

    def rectangle_frame(self, left_, top_, right_, bottom_,
                        rect_color=(222, 196, 176),
                        corner_color=(222, 196, 176),
                        length=25, thickness=4, rect_width=1):

        # rectangles are top, right, bottom, left
        cv2.rectangle(self.frame_, (left_, top_), (right_, bottom_), rect_color, rect_width)

        # top left
        cv2.line(self.frame_, (left_, top_), (left_ + length, top_), corner_color, thickness)
        cv2.line(self.frame_, (left_, top_), (left_, top_ + length), corner_color, thickness)

        # top right
        cv2.line(self.frame_, (right_, top_), (right_ - length, top_), corner_color, thickness)
        cv2.line(self.frame_, (right_, top_), (right_, top_ + length), corner_color, thickness)

        # bottom left
        cv2.line(self.frame_, (left_, bottom_), (left_ + length, bottom_), corner_color, thickness)
        cv2.line(self.frame_, (left_, bottom_), (left_, bottom_ - length), corner_color, thickness)

        # bottom right
        cv2.line(self.frame_, (right_, bottom_), (right_ - length, bottom_), corner_color, thickness)
        cv2.line(self.frame_, (right_, bottom_), (right_, bottom_ - length), corner_color, thickness)

        return self.frame_

    def face_recognition_(self):

        _, images_name_ = self.classify_Images()

        known_list_names = self.encodings_Images()
        print('| Encoding Successfully...')

        face_locations = face_recognition.face_locations(self.frame_)
        face_encodings = face_recognition.face_encodings(self.frame_, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_list_names, face_encoding)
            name = "Unknown".upper()
            face_distances = face_recognition.face_distance(known_list_names, face_encoding)
            best_match = np.argmin(face_distances)
            confidence = '-.-%'

            if matches[best_match]:
                name = images_name_[best_match].upper()
                confidence = self.face_confidence(face_distances[best_match])

                frame_ = self.rectangle_frame(left, top, right, bottom)

                cv2.putText(frame_, '| [ {} ] |'.format(name), (left, top - 20),
                            cv2.FONT_HERSHEY_PLAIN, 1.8,
                            (144, 128, 112), 2)
                cv2.putText(frame_, '{}'.format(confidence), (left, top - 50),
                            cv2.FONT_HERSHEY_PLAIN, 1.8,
                            (144, 128, 112), 2)
            else:
                frame_ = self.rectangle_frame(left, top, right, bottom,
                                              rect_color=(0, 0, 255), corner_color=(123, 104, 238))

                cv2.putText(frame_, '| [ {} ] |'.format(name), (left, top - 20),
                            cv2.FONT_HERSHEY_PLAIN, 1.8,
                            (123, 104, 238), 2)
                cv2.putText(frame_, '{}'.format(confidence), (left, top - 50),
                            cv2.FONT_HERSHEY_PLAIN, 1.8,
                            (123, 104, 238), 2)
        return self.frame_
