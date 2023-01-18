import time

import cv2
from modules.face_detection_module import FaceDetection
from modules.face_landmark_module import FaceLandmark
from modules.face_recognition_module import FaceRecognition
from modules.lips_detection_module import LipsDetection
from modules.iris_tracking_module import IrisTracking
from modules.head_pose_estimation_module import HeadPoseEstimation

from modules.pillow.custom_font_style import CustomFont

# from modules.pillow.custom_rectangle_frame import CustomFrame

cap = cv2.VideoCapture(0)

pTime = 0

# model = {'detection_obj': detection_obj,
#          'landmark_obj': landmark_obj,
#          'lips_obj': lips_obj,
#          'iris_obj': iris_obj,
#          'head_pose_obj': head_pose_obj
#          }
#
# file = open('main.pickle', "wb")
# file.write(pickle.dumps(model))
# file.close()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    while True:
        _, frame = cap.read()

        font_obj = CustomFont(frame)

        detection_obj = FaceDetection(frame)
        landmark_obj = FaceLandmark(frame)
        # lips_obj = LipsDetection(frame)
        iris_obj = IrisTracking(frame)
        head_pose_obj = HeadPoseEstimation(frame)

        frame_, no_of_face = detection_obj.face_detection()
        landmark_obj.face_landmark()
        iris_obj.iris_tracking()
        # lips_obj.lips_detection()
        head_pose_obj.head_pose_estimation()

        # recognition_obj = FaceRecognition(frame)

        # frame = recognition_obj.face_recognition_()

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # cv2.putText(frame, f'fps: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (237, 149, 100), 2)

        # cv2.putText(frame, 'No. of face | {} '.format(no_of_face), (20, 100), cv2.FONT_HERSHEY_PLAIN, 1.8,
        #             (219, 112, 147), 2)

        frame_text = font_obj.multiline_text(x=20, y=350, text_='fps | {} '
                                                                '\nFace No. | {}'.format(int(fps), no_of_face),
                                             fonts_style='fonts/Offside-Regular.ttf',
                                             fonts_size=30, text_color='#7B68EE', text_height_spacing=18)

        resize = cv2.resize(frame_text, (640, 480))

        cv2.imshow("| [ Exam hall monitering system ] |", resize)

        cv2.imwrite("output/main.jpg", resize)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
