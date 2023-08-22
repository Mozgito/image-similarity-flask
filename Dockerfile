FROM python:3.8.10-slim

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        libatlas-base-dev gfortran nginx supervisor libgl1 ffmpeg libsm6 libxext6

COPY ./requirements.txt /app/requirements.txt
RUN pip3 install --index-url http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    --no-cache-dir -r /app/requirements.txt

RUN useradd --no-create-home nginx

RUN rm /etc/nginx/sites-enabled/default

ENV STATIC_URL /static
ENV STATIC_PATH /app/static
ENV PYTHONPATH /app
ENV PYTHONBUFFERED 1

COPY server/ /etc/

COPY ./app /app
RUN mkdir /app/static/images && \
    mkdir -p /app/static/compare_results/original_images/350 && \
    mkdir -p /app/static/compare_results/original_images/700 && \
    mkdir -p /app/static/compare_results/data && \
    mkdir -p /app/static/compare_results/exchange_rate
WORKDIR /app
RUN chown -R nginx:nginx /app/static/compare_results

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisord.conf"]
