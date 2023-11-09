FROM python:3.9-slim

ENV DEEPLASIA_THREADS=4

# If your company uses a self-signed CA:
# ENV PIP_TRUSTED_HOST=download.pytorch.org

WORKDIR /app

COPY requirements.txt /app/.
RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8080

CMD [ "waitress-serve", "app:app"]
