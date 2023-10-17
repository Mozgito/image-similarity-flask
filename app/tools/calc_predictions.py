import numpy as np
import os
import time
from bson.binary import Binary
from database import DatabaseHandler
from dotenv import load_dotenv
from model import load_models
from pickle import dumps as pickle_dumps

load_dotenv()
np.seterr(divide='ignore', invalid='ignore')
db_handler = DatabaseHandler(os.environ.get("APP_MONGO_URL"), os.environ.get("APP_MONGO_DB"))


def update_prediction(item):
    query = {"image": item["image"], "site": item["site"]}
    result = db_handler.find_one_and_update('predictions', query, item)

    if result is None:
        db_handler.insert_one('predictions', item)


def is_file_older_than_x_days(file, days=1):
    file_time = os.path.getmtime(file)
    return (time.time() - file_time) / 3600 > 24 * days


if __name__ == '__main__':
    collection = 'bags'
    sites = db_handler.distinct(collection, 'site')

    for site in sites:
        images_path = 'images/{}/{}/'.format(collection, site)

        for image_name in os.listdir(images_path):
            image_path = os.path.join(images_path, image_name)

            if is_file_older_than_x_days(image_path):
                continue

            img_resnet, img_vgg16_flatten, img_vgg16_fc2 = load_models(image_path)
            binary_resnet = Binary(pickle_dumps(img_resnet, protocol=2))
            binary_vgg16_flatten = Binary(pickle_dumps(img_vgg16_flatten, protocol=2))
            binary_vgg16_fc2 = Binary(pickle_dumps(img_vgg16_fc2, protocol=2))

            img_item = {
                'image': image_name,
                'site': site,
                'type': collection,
                'resnet-avg-pool': binary_resnet,
                'vgg16-flatten': binary_vgg16_flatten,
                'vgg16-fc2': binary_vgg16_fc2
            }
            update_prediction(img_item)
