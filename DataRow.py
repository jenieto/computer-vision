class DataRow:
    def __init__(self, name, raw_images, processed_images, coordinates):
        self.name = name
        self.raw_images = raw_images
        self.processed_images = processed_images
        self.coordinates = coordinates
