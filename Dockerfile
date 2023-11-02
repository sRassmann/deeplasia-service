FROM python:3.9
# easier base image ?

COPY . /app

WORKDIR /app

RUN ["apt-get", "update"]
RUN ["apt-get", "-y", "install", "vim"]

#Install necessary packages from requirements.txt with no cache dir allowing for installation on machine with very little memory on board
RUN pip install -r requirements.txt

#Exposing the default waitress port
EXPOSE 8080

#Running the streamlit app
CMD ["python", "app.py"]