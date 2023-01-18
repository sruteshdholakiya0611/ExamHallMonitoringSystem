import time

import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

# initialize the width, height and fps of the video write object
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# initialize the FourCC and a video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter('../output/video/face-detection.mp4', fourcc, fps, (frame_width, frame_height))

pTime = 0

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(0.75)


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

    height, width, _ = frame.shape

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    result_face_detection = face_detection.process(rgb_image)

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

    cv2.imwrite("../output/face-detection.jpg", resize)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

output.release()
cap.release()
cv2.destroyAllWindows()
