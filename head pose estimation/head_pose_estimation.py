import cv2
import mediapipe as mp
import numpy as np
import time

cap = cv2.VideoCapture(0)

# initialize the width, height and fps of the video write object
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# initialize the FourCC and a video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter('../output/video/head-pose-estimation.mp4', fourcc, fps, (frame_width, frame_height))

pTime = 0

mp_face_mesh = mp.solutions.face_mesh
face_landmark = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                      refine_landmarks=True)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(0.75)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


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


while cap.isOpened():
    _, frame = cap.read()

    no_of_face = 0

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    result_face_landmarks = face_landmark.process(rgb_image)

    result_face_detection = face_detection.process(rgb_image)

    face_3d = []
    face_2d = []

    height, width, _ = frame.shape

    if result_face_landmarks.multi_face_landmarks:
        for facial_landmarks in result_face_landmarks.multi_face_landmarks:

            for source_index, target_index in mp_face_mesh.FACEMESH_IRISES:
                source = facial_landmarks.landmark[source_index]
                target = facial_landmarks.landmark[target_index]

                relative_source = (int(source.x * frame.shape[1]), int(source.y * frame.shape[0]))
                relative_target = (int(target.x * frame.shape[1]), int(target.y * frame.shape[0]))

                cv2.line(frame, relative_source, relative_target, (237, 149, 100), 1)
                cv2.circle(frame, relative_source, 1, (237, 149, 100), -1)
                cv2.circle(frame, relative_target, 1, (237, 149, 100), -1)

            for idx, landmarks in enumerate(facial_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    # if idx == 1:
                    #     nose_2d = (landmarks.x * width, landmarks.y * height)
                    #     nose_3d = (landmarks.x * width, landmarks.y * height, landmarks.z * 8000)

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

            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation_mat)

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
                text = "Looking | [ screen ]"

            condition = y < -10 or y > 10 or x > 10 or x < -10

            # Add the text on the image

            if condition:
                cv2.putText(frame, text, (20, 150), cv2.FONT_HERSHEY_PLAIN, 2,
                            (123, 104, 238), 2)
                cv2.putText(frame, "| [ ALERT ] |", (18, 450), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 2)
                cv2.imwrite("../output/head-pose-estimation-alert.jpg", frame)
            else:
                cv2.putText(frame, text, (20, 150), cv2.FONT_HERSHEY_PLAIN, 2, (238, 104, 123), 2)

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

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'fps: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (237, 149, 100), 2)

    cv2.putText(frame, 'No. of face | {} '.format(no_of_face), (20, 100), cv2.FONT_HERSHEY_PLAIN, 1.8,
                (219, 112, 147), 2)

    output.write(frame)

    resize = cv2.resize(frame, (640, 480))

    cv2.imshow("| [ Exam hall monitering system ] |", resize)

    cv2.imwrite("../output/head-pose-estimation.jpg", resize)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

output.release()
cap.release()
cv2.destroyAllWindows()
