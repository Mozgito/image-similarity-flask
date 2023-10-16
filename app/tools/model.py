import gc
import numpy as np
from keras.api._v2.keras.applications.resnet50 import ResNet50, preprocess_input
from keras.api._v2.keras.applications.vgg16 import VGG16
from keras.api._v2.keras.preprocessing import image as keras_image
from keras.api._v2.keras.models import Model
from keras.api._v2.keras.backend import clear_session

base_resnet50_model = ResNet50(weights='imagenet')
base_vgg16_model = VGG16(weights='imagenet')


def load_models(image_path):
    resnet_model = Model(inputs=base_resnet50_model.input, outputs=base_resnet50_model.get_layer('avg_pool').output)
    img_resnet = preprocess_and_predict(image_path, resnet_model)

    vgg16_flatten_model = Model(inputs=base_vgg16_model.input, outputs=base_vgg16_model.get_layer('flatten').output)
    img_vgg16_flatten = preprocess_and_predict(image_path, vgg16_flatten_model)

    vgg16_fc2_model = Model(inputs=base_vgg16_model.input, outputs=base_vgg16_model.get_layer('fc2').output)
    img_vgg16_fc2 = preprocess_and_predict(image_path, vgg16_fc2_model)

    return img_resnet, img_vgg16_flatten, img_vgg16_fc2


def preprocess_and_predict(image_path: str, model):
    image = keras_image.load_img(image_path, target_size=(224, 224))
    image = keras_image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    predictions = model.predict(image)
    clear_session()
    gc.collect()

    return predictions[0]
