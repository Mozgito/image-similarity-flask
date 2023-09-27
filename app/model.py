import gc
import numpy as np
from keras.api._v2.keras.applications.resnet50 import ResNet50, preprocess_input
from keras.api._v2.keras.applications.vgg16 import VGG16
from keras.api._v2.keras.preprocessing import image as keras_image
from keras.api._v2.keras.models import Model
from keras.api._v2.keras.backend import clear_session

base_resnet50_model = ResNet50(weights='imagenet')
base_vgg16_model = VGG16(weights='imagenet')


def get_resnet_model(layer='avg_pool'):
    return Model(inputs=base_resnet50_model.input, outputs=base_resnet50_model.get_layer(layer).output)


def get_vgg16_model(layer='flatten'):
    return Model(inputs=base_vgg16_model.input, outputs=base_vgg16_model.get_layer(layer).output)


def get_image_prediction(image_filepath: str, model):
    img = keras_image.load_img(image_filepath, target_size=(224, 224))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    clear_session()
    gc.collect()

    return features[0]
