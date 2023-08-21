## Release Notes

### Latest Changes

### 1.1.0
* Refactor compare data structure.
* Calculate similarity in background.
* Save calculation results in local storage.
* Remove SSIM metric, too slow.
* Auto-choose worker numbers for similarity pool.
* Move JS and CSS to local storage.

### 1.0.0
* This project (docker branch) is depending on the other `scrapy-spiders` project, that is scraping products (photos and data).
* Built on back of docker & docker-compose.
* Backend: Flask + uWSGI + Nginx + Supervisord + external Mongo. Frontend Bootstrap 5 + jQuery UI.
* Project will be ready on `port:8080` and it's under basic htpasswd authentication.
* Shares same network with scrapy Mongo.
* Image similarity is calculated by 4 metrics: PSNR, RMSE, SSIM and SRE. 
Top 10 results (by picture) from each metric by each site are chosen.