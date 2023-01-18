import os
import pickle

import cv2
import face_recognition

directory_name = '../../dataset'
directory_path = os.listdir(directory_name)


def classify_Images(directory_path_, directory_name_):
    dataset_images_ = []
    images_name_ = []

    for i, img in enumerate(directory_path_):
        print("[INFO] processing image {}/{}".format(i + 1, len(directory_path_)))
        current_image = cv2.imread(f'{directory_name_}/{img}')
        dataset_images_.append(current_image)
        images_name_.append(os.path.splitext(img)[0].capitalize())

    return dataset_images_, images_name_


print("[INFO] quantifying faces...")
dataset_images, images_name = classify_Images(directory_path_=directory_path, directory_name_=directory_name)
print('face names |', images_name)


def encodings_Images(dataset_images_):
    encode_list = []

    for img in dataset_images_:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode_image = face_recognition.face_encodings(img)[0]
        encode_list.append(encode_image)

    return encode_list


known_list_names = encodings_Images(dataset_images)
print('| Encoding Successfully...')

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
model = {"encodings": known_list_names, "names": images_name}

file = open('../../models/face recognition/encodings.pickle', "wb")
file.write(pickle.dumps(model))
file.close()
