import pandas as pd
import image_utils
import pickle
from DataRow import DataRow
import coordinates_reader
from ast import literal_eval as make_tuple

items_path = './data/anotaciones_itemsEvaluables_v3.csv'
quality_path = 'data/anotacionCalidadCopia.csv'
min_quality = 2
store_path = './data/data.pickle'


def filter_quality(data):
    quality_filter = data.quality >= min_quality
    return data[quality_filter]


def remove_spaces(data, keys=None):
    if keys is None:
        keys = data.keys()
    for key in keys:
        data[key] = data[key].apply(str.replace, args=(' ', ''))


def get_data():
    """
    Method to retrieve the data from the CSV files in items_path and quality_path filtered by min_quality
    :return: a pandas dataframe with a row per image and a column per figure
    """
    quality_data = pd.read_csv(quality_path, sep=';', names=['image', 'quality'], skipinitialspace=True)
    items_data = pd.read_csv(items_path, sep=';', names=['dir', 'image', 'figure', 'coords'], index_col=False, skipinitialspace=True)
    remove_spaces(items_data)
    remove_spaces(quality_data, keys=['image'])
    merged_data = pd.merge(items_data, quality_data, how='left', left_on='image', right_on='image')
    merged_data = filter_quality(merged_data)
    merged_data['figure'] = merged_data['figure'].str.replace('a', '')
    merged_data['figure'] = pd.to_numeric(merged_data['figure'])
    data = merged_data.drop(['dir', 'quality'], axis=1).sort_values(['image', 'figure'])
    data = data.groupby('image')['coords'].apply(lambda df: df.reset_index(drop=True)).unstack().reset_index()
    return data


def read_coordinates(row):
    coordinates = list(row[1:19])
    coordinates = list(map(lambda x: make_tuple(x) if isinstance(x, str) else None, coordinates))
    return coordinates_reader.transform_coordinates(coordinates)


def store_data(rows):
    with open(store_path, 'wb') as file:
        pickle.dump(rows, file)


def process_data():
    data = get_data()
    rows = []

    for index, row in data.iterrows():
        image_name = row['image']
        raw_images, processed_images = image_utils.read_images(image_name)
        coordinates = read_coordinates(row)
        row_item = DataRow(image_name, raw_images, processed_images, coordinates)
        rows.append(row_item)
    store_data(rows)


process_data()
