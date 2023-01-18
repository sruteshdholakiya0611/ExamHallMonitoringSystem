import os

import cv2
import face_recognition
import numpy as np
import time
import math

pTime = 0
directory_name = '../dataset'
directory_path = os.listdir(directory_name)


def classify_Images(directory_path_, directory_name_):
    dataset_images_ = []
    images_name_ = []

    for img in directory_path_:
        current_image = cv2.imread(f'{directory_name_}/{img}')
        dataset_images_.append(current_image)
        images_name_.append(os.path.splitext(img)[0].capitalize())

    return dataset_images_, images_name_


dataset_images, images_name = classify_Images(directory_path_=directory_path, directory_name_=directory_name)
print(images_name)


def encodings_Images(dataset_images_):
    encode_list = []

    for img in dataset_images_:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode_image = face_recognition.face_encodings(img)[0]
        encode_list.append(encode_image)

    return encode_list


known_list_names = encodings_Images(dataset_images)
print('| Encoding Successfully...')

cap = cv2.VideoCapture(0)


def face_confidence(face_distance, face_match_threshold=0.6):
    range_ = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_ * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'''


def rectangle_frame(image_, left_, top_, right_, bottom_,
                    rect_color=(222, 196, 176), corner_color=(222, 196, 176),
                    length=25, thickness=4, rect_width=1):

    # rectangles are top, right, bottom, left
    cv2.rectangle(image_, (left_, top_), (right_, bottom_), rect_color, rect_width)

    # top left
    cv2.line(image_, (left_, top_), (left_ + length, top_), corner_color, thickness)
    cv2.line(image_, (left_, top_), (left_, top_ + length), corner_color, thickness)

    # top right
    cv2.line(image_, (right_, top_), (right_ - length, top_), corner_color, thickness)
    cv2.line(image_, (right_, top_), (right_, top_ + length), corner_color, thickness)

    # bottom left
    cv2.line(image_, (left_, bottom_), (left_ + length, bottom_), corner_color, thickness)
    cv2.line(image_, (left_, bottom_), (left_, bottom_ - length), corner_color, thickness)

    # bottom right
    cv2.line(image_, (right_, bottom_), (right_ - length, bottom_), corner_color, thickness)
    cv2.line(image_, (right_, bottom_), (right_, bottom_ - length), corner_color, thickness)

    return image_


while True:
    _, frame = cap.read()

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_list_names, face_encoding)
        name = "Unknown".upper()
        face_distances = face_recognition.face_distance(known_list_names, face_encoding)
        best_match = np.argmin(face_distances)
        confidence = '-.-%'

        if matches[best_match]:
            name = images_name[best_match].upper()
            confidence = face_confidence(face_distances[best_match])

            frame = rectangle_frame(frame, left, top, right, bottom)

            cv2.putText(frame, '| [ {} ] |'.format(name), (left, top - 20),
                        cv2.FONT_HERSHEY_PLAIN, 1.8,
                        (144, 128, 112), 2)
            cv2.putText(frame, '{}'.format(confidence), (left, top - 50),
                        cv2.FONT_HERSHEY_PLAIN, 1.8,
                        (144, 128, 112), 2)
        else:
            frame = rectangle_frame(frame, left, top, right, bottom,
                                    rect_color=(0, 0, 255), corner_color=(123, 104, 238))

            cv2.putText(frame, '| [ {} ] |'.format(name), (left, top - 20),
                        cv2.FONT_HERSHEY_PLAIN, 1.8,
                        (123, 104, 238), 2)
            cv2.putText(frame, '{}'.format(confidence), (left, top - 50),
                        cv2.FONT_HERSHEY_PLAIN, 1.8,
                        (123, 104, 238), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 87, 51), 2)

    resize = cv2.resize(frame, (640, 480))

    cv2.imshow('Face Recognition', resize)

    cv2.imwrite("../output/face-recognition-webcam.jpg", resize)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
