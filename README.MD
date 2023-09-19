[![License](https://img.shields.io/badge/license-Apache-blue.svg)]()
[![Python](https://img.shields.io/badge/python->=3.8-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-greeb.svg)](https://docs.docker.com/)
# image-similarity-flask
Flask app for getting the similar products by photo. 
List of products and photos is coming from scraping the websites. 
Images are stored in directory of scraper and product data is stored in MongoDB.
### Features
* This project is depending on the other `scrapy-spiders` project, 
that is scraping products (photos and data).
* Built on back of docker & docker compose v2.
* Makefile for easy use.
* Backend: Flask + uWSGI + Nginx + Supervisord + external Mongo. Frontend Bootstrap 5 + jQuery UI.
* Project will be ready on `port:8080` and it's under basic htpasswd authentication.
* Shares same network with scrapy Mongo.
* Image similarity is calculated by 3 metrics: PSNR, RMSE, and SRE. 
Top 10 results (by picture) from each metric by each site are chosen.
### First-time setup
1. Set Mongo connection data and path to images in `.env`. 
Settings are equivalent to the scrapy project. 
Note: IP address of Mongo will be equal to Mongo container name.
2. Set Exchangerate API key.
3. Add user for Nginx authentication. More details in the README at `server/nginx/auth`.
4. Build project `make build` and start `make start`.
### TODO
- [ ] Replace Processing with Celery task manager.
- [ ] Improve similarity calculation with Tensorflow+Keras.
- [ ] Drop-down list with collections in DB.
### License
This project is licensed under the terms of the Apache license.
