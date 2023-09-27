import cv2 as cv
import heapq
import math
import numpy as np
import os
import pickle
import pymongo
import requests
import time
from flask import Flask, json, render_template, request, Response, redirect, send_from_directory, url_for
from PIL import Image
from scipy.spatial.distance import cosine
from tools import model
from threading import Thread

application = Flask(__name__)
application.config.from_prefixed_env('APP')
np.seterr(divide='ignore', invalid='ignore')
ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp', 'bmp']


def get_db():
    client = pymongo.MongoClient(application.config['MONGO_URL'], serverSelectionTimeoutMS=10000, connect=False)
    return client[application.config['MONGO_DB']]


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def resize_image(img_name: str) -> None:
    img = cv.imread(os.path.join(application.config['ORIG_IMAGES'], img_name), cv.IMREAD_COLOR)
    height = img.shape[0]
    width = img.shape[1]
    if height > width:
        preferred_height = height
        preferred_width = height
    else:
        preferred_height = width
        preferred_width = width

    pad_top = 0
    pad_bot = 0
    pad_left = 0
    pad_right = 0

    if height > width:
        preferred_width = round(preferred_height / height * width)
        pad_left = math.floor((preferred_height - preferred_width) / 2)
        pad_right = math.ceil((preferred_height - preferred_width) / 2)

    if height < width:
        preferred_height = round(preferred_width / width * height)
        pad_top = math.floor((preferred_width - preferred_height) / 2)
        pad_bot = math.ceil((preferred_width - preferred_height) / 2)

    if height != preferred_height or width != preferred_width:
        img_new = cv.resize(img, (preferred_width, preferred_height))
        img_new_padded = cv.copyMakeBorder(
            img_new,
            pad_top,
            pad_bot,
            pad_left,
            pad_right,
            cv.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
        cv.imwrite(os.path.join(application.config['ORIG_IMAGES'], img_name), img_new_padded)


def get_similar_by_metric(compare_results: dict) -> set:
    result = set()

    for metric, metric_values in compare_results.items():
        if metric == 'vgg16-fc2':
            result.update(set(heapq.nlargest(15, metric_values, key=metric_values.get)))
        else:
            result.update(set(heapq.nlargest(10, metric_values, key=metric_values.get)))

    return result


def calculate_similarity(img_resnet, img_vgg16_flatten, img_vgg16_fc2, collection: str, site: str) -> dict:
    result = {'resnet-avg-pool': {}, 'vgg16-flatten': {}, 'vgg16-fc2': {}}
    imgs_comp = get_db()['predictions'].find({'site': site, 'type': collection})

    for img_comp in imgs_comp:
        result['resnet-avg-pool'].update(
            {img_comp['image']: cosine_similarity(img_resnet, pickle.loads(img_comp['resnet-avg-pool']))}
        )
        result['vgg16-flatten'].update(
            {img_comp['image']: cosine_similarity(img_vgg16_flatten, pickle.loads(img_comp['vgg16-flatten']))}
        )
        result['vgg16-fc2'].update(
            {img_comp['image']: cosine_similarity(img_vgg16_fc2, pickle.loads(img_comp['vgg16-fc2']))}
        )

    return result


def cosine_similarity(a, b):
    return 1 - cosine(a, b)


def get_exchange_rate_data(currency='PHP') -> dict:
    if not is_dump_exist(application.config['EXCHANGE_RATE'], currency.lower() + '_rate'):
        update_exchange_rate_data(currency)

    rate_data = get_dump_data(application.config['EXCHANGE_RATE'], currency.lower() + '_rate')

    if rate_data['time_next_update_unix'] <= int(time.time()) and update_exchange_rate_data(currency):
        rate_data = get_dump_data(application.config['EXCHANGE_RATE'], currency.lower() + '_rate')

    return rate_data


def update_exchange_rate_data(currency='PHP') -> bool:
    rate_url = 'https://v6.exchangerate-api.com/v6/{}/latest/{}' \
        .format(application.config['EXCHANGE_RATE_APIKEY'], currency.upper())
    response = requests.get(rate_url)
    rate_data = json.loads(response.content)

    if 'result' in rate_data and rate_data['result'] == 'success':
        save_dump_data(application.config['EXCHANGE_RATE'], currency.lower() + '_rate', rate_data)
        return True

    return False


def is_dump_exist(path: str, dump_name: str) -> bool:
    return os.path.exists('{}/{}.json'.format(path, dump_name))


def get_dump_data(path: str, dump_name: str):
    with open('{}/{}.json'.format(path, dump_name), 'r') as f:
        data = json.load(f)

    return data


def save_dump_data(path: str, dump_name: str, dump_data) -> None:
    with open('{}/{}.json'.format(path, dump_name), 'w') as f:
        json.dump(dump_data, f)


def remove_dump_data(path: str, dump_name: str) -> None:
    os.remove('{}/{}.json'.format(path, dump_name))


@application.route('/original_images/<filename>')
def get_original_image(filename) -> Response:
    return send_from_directory(application.config['ORIG_IMAGES'], filename)


@application.route('/')
def index() -> str:
    return render_template('base.html', title='Main')


@application.route('/upload-image', methods=['POST'])
def upload_image() -> Response:
    file = request.files['origImgUpload']

    if file and allowed_file(file.filename):
        filename = str(int(time.time())) + '.png'
        img = Image.open(file)
        img.save(os.path.join(application.config['ORIG_IMAGES'], filename))
        resize_image(filename)

        return redirect(url_for('similarity', original_image=filename))

    return Response(status=204)


@application.route('/all-products')
def all_products() -> str:
    return render_template('all_products.html', title='All products')


@application.route('/similarity')
def similarity() -> [str, Response]:
    image_name = request.args.get('original_image')
    recalculate = request.args.get('recalc')

    if not isinstance(image_name, str) or not allowed_file(image_name):
        return Response(status=400)

    if not os.path.isfile(os.path.join(application.config['ORIG_IMAGES'], image_name)):
        return Response(status=404)

    if recalculate:
        remove_dump_data(application.config['COMPARE_DATA'], image_name + '_log')

    if not is_dump_exist(application.config['COMPARE_DATA'], image_name + '_log'):
        Thread(target=api_similarity_calculate, name="Downloader", args=[image_name]).start()

        return render_template(
            'similar_products.html',
            title='Similar products',
            original_image=image_name,
            calculate_status='calculating'
        )

    similarity_status = get_dump_data(application.config['COMPARE_DATA'], image_name + '_log')
    if similarity_status == 'finished':
        data = json.dumps({'data': get_dump_data(application.config['COMPARE_DATA'], image_name)})
        return render_template(
            'similar_products.html',
            title='Similar products',
            original_image=image_name,
            calculate_status=similarity_status,
            product_data=data
        )
    else:
        return render_template(
            'similar_products.html',
            title='Similar products',
            original_image=image_name,
            calculate_status=similarity_status
        )


@application.route('/results')
def results() -> str:
    table_data = []

    for image_name in os.listdir(application.config['ORIG_IMAGES']):
        image_path = os.path.join(application.config['ORIG_IMAGES'], image_name)
        status_path = os.path.join(application.config['COMPARE_DATA'], image_name + '_log.json')

        if os.path.isfile(image_path) and os.path.isfile(status_path):
            table_data.append({
                'image': image_path,
                'date': time.ctime(os.path.getmtime(status_path)),
                'url': url_for('similarity', original_image=image_name),
                'calculate_url': url_for('similarity', original_image=image_name, recalc=True),
                'status': get_dump_data(application.config['COMPARE_DATA'], image_name + '_log')
            })

    return render_template(
        'results.html',
        title='Results',
        results_data=table_data
    )


@application.route('/api/all-products')
def api_all_products() -> Response:
    collection = 'bags'
    php_rate = get_exchange_rate_data()['conversion_rates']
    table_data = []

    for row in get_db()[collection].find():
        if row['currency'] != 'PHP':
            php_price = round(float(row['price']) / php_rate[row['currency']], 2)
        else:
            php_price = row['price']

        table_data.append({
            'image': row['images'][0]['path'] if len(row['images']) > 0 else '',
            'name': row['name'],
            'price': row['price'],
            'currency': row['currency'],
            'php_price': php_price,
            'url': row['url'],
            'site': row['site']
        })

    return json.jsonify({'data': table_data})


@application.route('/api/similarity/<image_name>/calculate')
def api_similarity_calculate(image_name: str) -> Response:
    with application.app_context():
        if not isinstance(image_name, str) or not allowed_file(image_name):
            return Response(status=400)

        if not os.path.isfile(os.path.join(application.config['ORIG_IMAGES'], image_name)):
            return Response(status=404)

        try:
            collection = 'bags'
            sites = get_db()['predictions'].distinct('site')
            php_rate = get_exchange_rate_data()['conversion_rates']
            table_data = []
            save_dump_data(application.config['COMPARE_DATA'], image_name + '_log', 'calculating')
            img_resnet = model.get_image_prediction(
                os.path.join(application.config['ORIG_IMAGES'], image_name),
                model.get_resnet_model()
            )
            img_vgg16_flatten = model.get_image_prediction(
                os.path.join(application.config['ORIG_IMAGES'], image_name),
                model.get_vgg16_model()
            )
            img_vgg16_fc2 = model.get_image_prediction(
                os.path.join(application.config['ORIG_IMAGES'], image_name),
                model.get_vgg16_model('fc2')
            )

            for site in sites:
                total_similarity_result = calculate_similarity(
                    img_resnet,
                    img_vgg16_flatten,
                    img_vgg16_fc2,
                    collection,
                    site
                )
                top_similar_images = get_similar_by_metric(total_similarity_result)

                for similar_image_name in top_similar_images:
                    similar_image_path = '{}/{}/{}'.format(collection, site, similar_image_name)
                    product_data = get_db()[collection].find({'images.path': similar_image_path}).sort([('price', 1)]).limit(2)

                    for row in product_data:
                        if next(filter(lambda d: d.get('url') == row['url'], table_data), None) is None:
                            if row['currency'] != 'PHP':
                                php_price = round(float(row['price']) / php_rate[row['currency']], 2)
                            else:
                                php_price = row['price']

                            table_data.append({
                                'image': similar_image_path,
                                'name': row['name'],
                                'price': row['price'],
                                'currency': row['currency'],
                                'php_price': php_price,
                                'url': row['url'],
                                'site': row['site']
                            })

            save_dump_data(application.config['COMPARE_DATA'], image_name, table_data)
            save_dump_data(application.config['COMPARE_DATA'], image_name + '_log', 'finished')

            return json.jsonify({'data': table_data})
        except Exception as e:
            save_dump_data(application.config['COMPARE_DATA'], image_name + '_log', str(e))
            return Response(status=400, response=str(e))


@application.route('/api/similarity/<image_name>/result')
def api_similarity_result(image_name: str) -> Response:
    if is_dump_exist(application.config['COMPARE_DATA'], image_name):
        return json.jsonify({'data': get_dump_data(application.config['COMPARE_DATA'], image_name)})

    return Response(status=404)
