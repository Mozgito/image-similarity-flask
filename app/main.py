import cv2 as cv
import heapq
import math
import numpy as np
import os
import pymongo
import requests
import time
from flask import Flask, json, render_template, request, Response, redirect, send_from_directory, url_for
from image_similarity_measures.quality_metrics import psnr, rmse, sre
from multiprocessing import Pool, Process, cpu_count
from PIL import Image

application = Flask(__name__)
application.config.from_prefixed_env('APP')
np.seterr(divide='ignore', invalid='ignore')
ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp', 'bmp']


def get_db():
    client = pymongo.MongoClient(application.config['MONGO_URL'], serverSelectionTimeoutMS=10000, connect=False)
    return client[application.config['MONGO_DB']]


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def resize_image(img_name: str, size: int) -> None:
    img = cv.imread(os.path.join(application.config['ORIG_IMAGES'], img_name), cv.IMREAD_COLOR)
    height = img.shape[0]
    width = img.shape[1]
    preferred_height = size
    preferred_width = size
    pad_top = 0
    pad_bot = 0
    pad_left = 0
    pad_right = 0

    if height > width:
        preferred_width = round(preferred_height / height * width)
        pad_left = math.floor((size - preferred_width) / 2)
        pad_right = math.ceil((size - preferred_width) / 2)

    if height < width:
        preferred_height = round(preferred_width / width * height)
        pad_top = math.floor((size - preferred_height) / 2)
        pad_bot = math.ceil((size - preferred_height) / 2)

    if height != size or width != size:
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
        cv.imwrite(os.path.join(application.config['ORIG_IMAGES'], str(size), img_name), img_new_padded)

    if height == size and width == size:
        cv.imwrite(os.path.join(application.config['ORIG_IMAGES'], str(size), img_name), img)


def get_resized_image_path(original_img_name: str, site: str) -> str:
    if site in ['Lazada', 'Shopee', 'Temu']:
        return os.path.join(application.config['ORIG_IMAGES'], '700', original_img_name)

    return os.path.join(application.config['ORIG_IMAGES'], '350', original_img_name)


def compare_images(original_img_path, compare_img_path, compare_img_name: str) -> dict:
    original_img = cv.imread(original_img_path)
    compare_img = cv.imread(compare_img_path)

    return {
        'img_name': compare_img_name,
        'psnr': psnr(original_img, compare_img),
        'rmse': rmse(original_img, compare_img),
        'sre': sre(original_img, compare_img)
    }


def get_similar_by_metric(compare_results: dict, metric_size=10) -> set:
    result = set()

    for metric, metric_values in compare_results.items():
        if metric == 'rmse':
            result.update(set(heapq.nsmallest(metric_size, metric_values, key=metric_values.get)))
        else:
            result.update(set(heapq.nlargest(metric_size, metric_values, key=metric_values.get)))

    return result


def calculate_similarity(original_img_path: str, collection: str, site: str) -> dict:
    total_result = {'psnr': {}, 'rmse': {}, 'sre': {}}
    with Pool(processes=cpu_count()) as pool:
        compare_path = '{}/{}/{}/'.format(application.config['COMPARE_IMAGES'], collection, site)
        args = [
            (original_img_path,
             os.path.join(compare_path, compare_img_name),
             compare_img_name)
            for compare_img_name in os.listdir(compare_path)
        ]
        result = pool.starmap_async(compare_images, args)

        for value in result.get():
            total_result['psnr'].update({value['img_name']: value['psnr']})
            total_result['rmse'].update({value['img_name']: value['rmse']})
            total_result['sre'].update({value['img_name']: value['sre']})
        pool.close()
        pool.join()

    return total_result


def get_exchange_rate_data(currency='PHP') -> dict:
    if not is_dump_exist('static/compare_results', currency.lower() + '_rate'):
        update_exchange_rate_data(currency)

    rate_data = get_dump_data('static/compare_results', currency.lower() + '_rate')

    if rate_data['time_next_update_unix'] <= int(time.time()) and update_exchange_rate_data(currency):
        rate_data = get_dump_data('static/compare_results', currency.lower() + '_rate')

    return rate_data


def update_exchange_rate_data(currency='PHP') -> bool:
    rate_url = 'https://v6.exchangerate-api.com/v6/{}/latest/{}' \
        .format(application.config['EXCHANGE_RATE_APIKEY'], currency.upper())
    response = requests.get(rate_url)
    rate_data = json.loads(response.content)

    if 'result' in rate_data and rate_data['result'] == 'success':
        save_dump_data('static/compare_results', currency.lower() + '_rate', rate_data)
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
        resize_image(filename, 700)
        resize_image(filename, 350)

        return redirect(url_for('similarity', original_image=filename))

    return Response(status=204)


@application.route('/all-products')
def all_products() -> str:
    return render_template('all_products.html', title='All products')


@application.route('/similarity')
def similarity() -> [str, Response]:
    image_name = request.args.get('original_image')

    if not isinstance(image_name, str) or not allowed_file(image_name):
        return Response(status=400)

    if not os.path.isfile(os.path.join(application.config['ORIG_IMAGES'], image_name)):
        return Response(status=404)

    if not is_dump_exist(application.config['COMPARE_DATA'], image_name + '_log'):
        Process(target=api_similarity_calculate, args=[image_name]).start()

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
        if os.path.isfile(image_path):
            table_data.append({
                'image': image_path,
                'date': time.ctime(os.path.getmtime(image_path)),
                'url': url_for('similarity', original_image=image_name),
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
    if not isinstance(image_name, str) or not allowed_file(image_name):
        return Response(status=400)

    if not os.path.isfile(os.path.join(application.config['ORIG_IMAGES'], image_name)):
        return Response(status=404)

    try:
        collection = 'bags'
        sites = get_db()[collection].distinct('site')
        php_rate = get_exchange_rate_data()['conversion_rates']
        top_similar_images = set()
        table_data = []
        save_dump_data(application.config['COMPARE_DATA'], image_name + '_log', 'calculating')

        for site in sites:
            original_img_path = get_resized_image_path(image_name, site)
            total_similarity_result = calculate_similarity(original_img_path, collection, site)
            top_similar_images.update(get_similar_by_metric(total_similarity_result, 25))

            for similar_image_name in top_similar_images:
                similar_image_path = '{}/{}/{}'.format(collection, site, similar_image_name)
                product_data = get_db()[collection].find({'images.path': similar_image_path}).sort([('price', 1)]).limit(1)

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
