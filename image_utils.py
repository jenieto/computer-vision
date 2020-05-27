import cv2
import os
import numpy as np
from sklearn import preprocessing


IMAGE_CHANNELS = 3
IMAGE_SIZE = (128, 128, IMAGE_CHANNELS)
images_path = 'data/images/'


def read_images(image_name):
    path = os.path.join(images_path, 'grafo_' + image_name + '.png')
    raw_X = cv2.imread(path, 0)
    proc_X = None
    if raw_X is None:
        return None, None
    shape = raw_X.shape
    raw_X_1 = raw_X[:shape[0] // 2, :shape[1] // 2]  # Imagen original
    raw_X_2 = raw_X[:shape[0] // 2, shape[1] // 2:]  # Imagen original invertida
    raw_X_3 = raw_X[shape[0] // 2:, :shape[1] // 2]  # Imagen de grafos detalle alto
    raw_X_4 = raw_X[shape[0] // 2:, shape[1] // 2:]  # Imagen de grafos detalle bajo

    raw_X_1 = cv2.resize(raw_X_1, dsize=(IMAGE_SIZE[0], IMAGE_SIZE[1]))
    raw_X_2 = cv2.resize(raw_X_2, dsize=(IMAGE_SIZE[0], IMAGE_SIZE[1]))
    raw_X_3 = cv2.resize(raw_X_3, dsize=(IMAGE_SIZE[0], IMAGE_SIZE[1]))
    raw_X_4 = cv2.resize(raw_X_4, dsize=(IMAGE_SIZE[0], IMAGE_SIZE[1]))

    scaler = preprocessing.StandardScaler()
    proc_X_1 = scaler.fit_transform(raw_X_1)  # Opción más simple: raw_X_1 / 255
    proc_X_2 = scaler.fit_transform(raw_X_2)  # Opción más simple: raw_X_2 / 255
    proc_X_3 = scaler.fit_transform(raw_X_3)  # Opción más simple: raw_X_3 / 255
    proc_X_4 = scaler.fit_transform(raw_X_4)  # Opción más simple: raw_X_4 / 255
    if IMAGE_CHANNELS == 1:
        raw_X = raw_X_2  # Sólo se usa la imagen original invertida
        proc_X = proc_X_2
    elif IMAGE_CHANNELS == 2:
        raw_X = np.stack((raw_X_2, raw_X_3), axis=-1)  # Se usa imagen original invertida + grafos alto detalle
        proc_X = np.stack((proc_X_2, proc_X_3), axis=-1)
    elif IMAGE_CHANNELS == 3:
        raw_X = np.stack((raw_X_2, raw_X_3, raw_X_4),
                         axis=-1)  # Se usa imagen original invertida + grafos alto detalle + grafos bajo detalle
        proc_X = np.stack((proc_X_2, proc_X_3, proc_X_4), axis=-1)
    raw_X = np.expand_dims(raw_X, axis=0)
    proc_X = np.expand_dims(proc_X, axis=0)
    return raw_X, proc_X


def show_image(image):
    cv2.imshow('Image', cv2.cvtColor(image[0, :, :, 0], cv2.COLOR_GRAY2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
