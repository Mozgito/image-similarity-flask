[![License](https://img.shields.io/badge/license-Apache-blue.svg)]()
[![Python](https://img.shields.io/badge/python->=3.8-blue.svg)](https://www.python.org/downloads/)
[![Python](https://img.shields.io/badge/Tensorflow->=2.8.0-orange.svg)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/docker-greeb.svg)](https://docs.docker.com/)
# image-similarity-flask
Flask app for getting the similar products by photo. Similarity is calculated by Tensorflow+Keras.
List of products and photos is coming from scraping the websites. 
Images are stored in directory of scraper and product data is stored in MongoDB.
Image predictions are also stored in same MongoDB.
### Features
* This project is depending on the other `scrapy-spiders` project, 
that is scraping products (photos and data).
* Built on back of docker & docker compose v2.
* Makefile for easy use.
* Backend: Flask + uWSGI + Nginx + Supervisord + external Mongo. Frontend Bootstrap 5 + jQuery UI.
* Project will be ready on `port:8080` and it's under basic htpasswd authentication.
* Shares same network with scrapy Mongo.
* Image similarity is calculated by 3 models: ResNet50 Avg-Pool, VGG16 FC2, and VGG16 Flatten. 
Top 10-15 results (by picture) from each model by each site are chosen.
### First-time setup
1. Set Mongo connection data and path to images in `.env`. 
Settings are equivalent to the scrapy project. 
Note: IP address of Mongo will be equal to Mongo container name.
2. Set Exchangerate API key.
3. Add user for Nginx authentication. More details in the README at `server/nginx/auth`.
4. Download Tensorflow wheel for Non-AVX and SSE2 CPU `wget https://tf.novaal.de/core2/tensorflow-2.8.0-cp38-cp38-linux_x86_64.whl`
Or use the wheel with needed instructions.
5. Pre-downloaded models' weights can be put into `models` folder.
6. Build project `make build` and start `make start`.
### TODO
- [ ] Replace Threading with Celery task manager.
- [ ] Drop-down list with collections in DB.
- [ ] Bestsellers statistics.
### License
This project is licensed under the terms of the Apache license.
