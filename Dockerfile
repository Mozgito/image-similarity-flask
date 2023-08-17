FROM python:3.8.10-slim

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        libatlas-base-dev gfortran nginx supervisor libgl1 ffmpeg libsm6 libxext6

RUN pip3 install uwsgi

COPY ./requirements.txt /app/requirements.txt
RUN pip3 install --index-url http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    --no-cache-dir -r /app/requirements.txt

RUN useradd --no-create-home nginx

RUN rm /etc/nginx/sites-enabled/default
RUN rm -r /root/.cache

ENV STATIC_URL /static
ENV STATIC_PATH /app/static
ENV PYTHONPATH=/app
ENV PYTHONBUFFERED=1

COPY server/auth/.htpasswd /etc/nginx/auth/
COPY server/nginx.conf /etc/nginx/
COPY server/uwsgi_params /etc/nginx/
COPY server/site.conf /etc/nginx/conf.d/
COPY server/uwsgi.ini /etc/uwsgi/
COPY server/supervisord.conf /etc/supervisor/

COPY ./app /app
RUN mkdir /app/static/images && mkdir -p /app/static/original_images/350 && mkdir -p /app/static/original_images/700
WORKDIR /app
RUN chown -R nginx:nginx /app/static/original_images

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
