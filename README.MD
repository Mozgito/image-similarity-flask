[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()
[![Python](https://img.shields.io/badge/python->=3.8-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-greeb.svg)](https://docs.docker.com/)
# image-similarity-flask
Flask app for getting the similar products by photo. List of products and photos are coming from scraping the websites. 
Images are stored in directory and product data stored in MongoDB.
### Features
* This project (docker branch) is depending on the other `scrapy-spiders` project, that is scraping products (photos and data).
* Built on back of docker & docker-compose.
* Backend: Flask + uWSGI + Nginx + Supervisord + external Mongo. Frontend Bootstrap 5 + jQuery UI.
* Project will be ready on `port:8080` and it's under basic htpasswd authentication.
* Shares same network with scrapy Mongo.
* Image similarity is calculated by 4 metrics: PSNR, RMSE, SSIM and SRE. 
Top 10 results (by picture) from each metric by each site are chosen.
### First-time setup
1. Set Mongo connection data and path to images in `.env`. Settings are equivalent to the scrapy project. 
Note: IP address of Mongo will be equal to Mongo container name.
2. Add user for Nginx authentication. More details in the README at `server/auth`.
3. Build and start project `docker compose up -d`
### TODO
- [ ] If comparison process takes more than 60s, AJAX will return 504.
Comparison has to be done in background. Check Celery task manager.
- [ ] Controlling the comparison process. If page is refreshed or new comparison started, kill the current process. 
Or save the result in file that later can be browsed or downloaded.
- [ ] PH Pesos exchange rate (to USD, CNY) and column with converted price for products.
- [ ] Selector for similarity metrics.
- [ ] Drop-down list with collections in DB.
- [x] Auto-choose number of Pool workers according to the number of CPU cores.
### License
This project is licensed under the terms of the Apache license.
