FROM python:3.9

COPY . /app

WORKDIR /app

RUN ["apt-get", "update"]
RUN ["apt-get", "-y", "install", "vim"]

#Install necessary packages from requirements.txt with no cache dir allowing for installation on machine with very little memory on board
RUN pip install -r requirements.txt

#Exposing the default streamlit port
EXPOSE 8080

#Running the streamlit app
ENTRYPOINT ["streamlit", "run", "--server.maxUploadSize=20", "--server.port=8080"]
CMD ["main.py"]