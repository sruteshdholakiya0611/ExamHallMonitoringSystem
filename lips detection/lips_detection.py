import time

import cv2
import dlib
import mediapipe as mp
from scipy.spatial import distance

cap = cv2.VideoCapture(0)

# initialize the width, height and fps of the video write object
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# initialize the FourCC and a video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter('../output/video/lips-detection.mp4', fourcc, fps, (frame_width, frame_height))

pTime = 0

LIPS = [60, 62, 64, 66]

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(0.75)

mp_face_mesh = mp.solutions.face_mesh
face_landmark = mp_face_mesh.FaceMesh(max_num_faces=100, refine_landmarks=True,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../models/shape_predictor_68_face_landmarks.dat")


def rectangle_frame(image, rectangle_box, length=25, thickness=4, rect_width=1):
    left, top, right, bottom = rectangle_box
    x1, y1 = left + right, top + bottom

    # rectangles are left, top, right, bottom ...
    cv2.rectangle(image, rectangle_box, (222, 196, 176), rect_width)

    # top left
    cv2.line(image, (left, top), (left + length, top), (222, 196, 176), thickness)
    cv2.line(image, (left, top), (left, top + length), (222, 196, 176), thickness)

    # top right
    cv2.line(image, (x1, top), (x1 - length, top), (222, 196, 176), thickness)
    cv2.line(image, (x1, top), (x1, top + length), (222, 196, 176), thickness)

    # bottom left
    cv2.line(image, (left, y1), (left + length, y1), (222, 196, 176), thickness)
    cv2.line(image, (left, y1), (left, y1 - length), (222, 196, 176), thickness)

    # bottom right
    cv2.line(image, (x1, y1), (x1 - length, y1), (222, 196, 176), thickness)
    cv2.line(image, (x1, y1), (x1, y1 - length), (222, 196, 176), thickness)

    return image


def get_mouth_ratio(mouth_points, face_landmarks):
    left = (face_landmarks.part(mouth_points[0]).x, face_landmarks.part(mouth_points[0]).y)  # L1
    right = (face_landmarks.part(mouth_points[2]).x, face_landmarks.part(mouth_points[2]).y)  # L5
    top = (face_landmarks.part(mouth_points[1]).x, face_landmarks.part(mouth_points[1]).y)  # L3
    bottom = (face_landmarks.part(mouth_points[3]).x, face_landmarks.part(mouth_points[3]).y)  # L7

    dist1 = distance.euclidean(top, bottom)
    dist2 = distance.euclidean(left, right)

    ratio = float(dist1 / dist2)

    return ratio


while True:
    _, frame = cap.read()

    no_of_face = 0

    height, width, _ = frame.shape

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result_face_landmarks = face_landmark.process(rgb_image)

    result_face_detection = face_detection.process(rgb_image)

    dlib_face_detection = detector(gray_image)

    if result_face_landmarks.multi_face_landmarks:
        for facial_landmarks in result_face_landmarks.multi_face_landmarks:

            for source_index, target_index in mp_face_mesh.FACEMESH_IRISES:
                source = facial_landmarks.landmark[source_index]
                target = facial_landmarks.landmark[target_index]

                relative_source = (int(source.x * frame.shape[1]), int(source.y * frame.shape[0]))
                relative_target = (int(target.x * frame.shape[1]), int(target.y * frame.shape[0]))

                cv2.line(frame, relative_source, relative_target, (192, 158, 125), 1)
                cv2.circle(frame, relative_source, 1, (192, 158, 125), -1)
                cv2.circle(frame, relative_target, 1, (192, 158, 125), -1)

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=facial_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())

    if result_face_detection.detections:
        for _, detection in enumerate(result_face_detection.detections):
            relative_bounding_box = detection.location_data.relative_bounding_box

            bbox = int(relative_bounding_box.xmin *
                       width), int(relative_bounding_box.ymin *
                                   height), int(relative_bounding_box.width *
                                                width), int(relative_bounding_box.height * height)

            frame = rectangle_frame(frame, bbox)
            cv2.putText(frame, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                        (222, 196, 176), 2)

            no_of_face = no_of_face + 1

        for face in dlib_face_detection:

            landmarks = predictor(gray_image, face)

            mouth_ratio = get_mouth_ratio(LIPS, landmarks)

            condition = (mouth_ratio > 0.1)

            if condition:
                cv2.putText(frame, '| [ Mouth is open! ] |', (20, 150), cv2.FONT_HERSHEY_PLAIN, 2,
                            (123, 104, 238), 2)
                cv2.putText(frame, "| [ ALERT ] |", (18, 450), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 2)
                cv2.imwrite("../output/lips-detection-alert.jpg", frame)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'fps: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (237, 149, 100), 2)

    cv2.putText(frame, 'No. of face | {} '.format(no_of_face), (20, 100), cv2.FONT_HERSHEY_PLAIN, 1.8,
                (219, 112, 147), 2)

    output.write(frame)

    resize = cv2.resize(frame, (640, 480))

    cv2.imshow("| [ Exam hall monitering system ] |", resize)

    cv2.imwrite("../output/lips-detection.jpg", resize)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

output.release()
cap.release()
cv2.destroyAllWindows()
