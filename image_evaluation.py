import pandas as pd
import cv2
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import csv

figure = 'a18'
items_path = './data/anotaciones_itemsEvaluables_v3.csv'
quality_path = './data/anotacionCalidadCopia.csv'
min_quality = 4
images_path = 'data/raw_images/'


def filter_quality(data):
    quality_filter = data.quality >= min_quality
    return data[quality_filter]


def remove_spaces(data, keys=None):
    if keys is None:
        keys = data.keys()
    for key in keys:
        data[key] = data[key].apply(str.replace, args=(' ', ''))


def read_image(name):
    path = os.path.join(images_path, name)
    return cv2.imread(path, 0)


def evaluate_image(index, total, image, points=[]):
    img_with_points = image
    for p in points:
        img_with_points = cv2.circle(img_with_points, (p[0], p[1]), 3, (0, 0, 255), -1)
    plt.imshow(img_with_points, cmap='gray')
    plt.show()
    x = input(f'Image {index}/{total}, is valid? (y/n): ')
    if x == 'y' or x == 'Y':
        return True
    return False


quality_data = pd.read_csv(quality_path, sep=';', names=['image', 'quality'], skipinitialspace=True)
items_data = pd.read_csv(items_path, sep=';', names=['dir', 'image', 'figure', 'coords'], index_col=False, skipinitialspace=True)
remove_spaces(items_data)
remove_spaces(quality_data, keys=['image'])
merged_data = pd.merge(items_data, quality_data, how='left', left_on='image', right_on='image')
merged_data = filter_quality(merged_data)
data = merged_data[merged_data['figure'] == figure]


with open(f'./data/data_figure_{figure}.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    total = data.shape[0]
    i = 1
    for index, row in data.iterrows():
        image_name = row['image']
        coords = row['coords']
        quality = int(row['quality'])
        flat = np.array([int(s) for s in re.findall('\d+', row['coords'])])  # Parsea los números en el texto
        mat = flat.reshape((-1, 2))  # Reagrupa las coordenadas en grupos de dos
        image = read_image(image_name)
        is_valid = evaluate_image(i, total, image, mat)
        writer.writerow(['REY_roi_manualSelection1', image_name, figure, coords, is_valid, quality])
        i = i+1
