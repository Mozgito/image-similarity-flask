import numpy as np
import pickle
import pymongo
import os
import time
from bson.binary import Binary
from dotenv import load_dotenv
from model import load_models

load_dotenv()
np.seterr(divide='ignore', invalid='ignore')


def get_db():
    client = pymongo.MongoClient(os.environ.get("APP_MONGO_URL"), serverSelectionTimeoutMS=10000, connect=False)
    return client[os.environ.get("APP_MONGO_DB")]


def update_prediction(db, item):
    if db['predictions'].find_one_and_update(
            {"image": item["image"], "site": item["site"]},
            {"$set": item}) is None:
        db['predictions'].insert_one(item)


def is_file_older_than_x_days(file, days=1):
    file_time = os.path.getmtime(file)
    return (time.time() - file_time) / 3600 > 24 * days


if __name__ == '__main__':
    database = get_db()
    collection = 'bags'
    sites = database[collection].distinct('site')

    for site in sites:
        images_path = 'images/{}/{}/'.format(collection, site)

        for image_name in os.listdir(images_path):
            image_path = os.path.join(images_path, image_name)

            if is_file_older_than_x_days(image_path):
                continue

            img_resnet, img_vgg16_flatten, img_vgg16_fc2 = load_models(image_path)
            binary_resnet = Binary(pickle.dumps(img_resnet, protocol=2))
            binary_vgg16_flatten = Binary(pickle.dumps(img_vgg16_flatten, protocol=2))
            binary_vgg16_fc2 = Binary(pickle.dumps(img_vgg16_fc2, protocol=2))

            img_item = {
                'image': image_name,
                'site': site,
                'type': collection,
                'resnet-avg-pool': binary_resnet,
                'vgg16-flatten': binary_vgg16_flatten,
                'vgg16-fc2': binary_vgg16_fc2
            }
            update_prediction(database, img_item)
