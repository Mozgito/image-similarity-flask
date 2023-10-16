## Release Notes

### Latest Changes
* SVG favicon
* Refactor "model" module

### 1.2.2
* Page with bestsellers with filter by report date

### 1.2.1
* Calculating image predictions tool

### 1.2.0
* Tensorflow + Keras similarity
* ResNet50 Avg-Pool, VGG16 FC2, and VGG16 Flatten models
* Update Dockerfile and requirements
* Recalculate button on result page

### 1.1.7
* Image resize fix
* Docker container restart policy
* Responsive layout, adjusted for mobile devices

### 1.1.5
* Makefile for easy use
* Add default sorting in tables (Site -> PHP price)
* Display error message if occurs during calculation
* Table rendering improvements

### 1.1.3
* Frontend: Word-break for links, Scroll top on changing products page.
* Correct ENV and app configuration.
* Add currency converter. Convert prices to PHP.
* Add page with all compare results.

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