# Deeplasia Service

Deeplasia is a prior-free deep learning approach to asses bone age in children and adolescents.
This repository contains a RESTfull service with a simple API to process X-ray images and predict bone age in months.

Please refer for more information:

* http://www.deeplasia.de/
* https://github.com/aimi-bonn/Deeplasia

[![Build Docker Image](https://github.com/CrescNet/deeplasia-service/actions/workflows/build.yml/badge.svg)](https://github.com/CrescNet/deeplasia-service/actions/workflows/build.yml)

## How to Use

In order to run this application, you must provide the deep learning models. Please contact use to get them.

Use the environment variable `DEEPLASIA_THREADS` to limit the number of threads used by [PyTorch](https://pytorch.org/) (defaults to 4 threads).

### Run in Conda Environment

**Requirements:**

* [Conda](https://docs.conda.io) must be installed
* Deep learning models are located in the directory `./models`

Run the following CLI commands and navigate to <http://localhost:5000/>.

```sh
conda create -n flask_ba python=3.9
conda activate flask_ba
pip install -r requirements.txt
python flask run 
```

### Run with Docker

**Requirements:**

* [Docker](https://docs.docker.com/engine/install/) must be installed
* Deep learning models are not included in the image and must be mounted on container start

You can use our pre built Docker image to run the application:

```sh
docker run -p 8080:8080 -v ./models:/app/models ghcr.io/crescnet/deeplasia-service
```

Or you can build the image yourself (clone this repository first):

```bash
docker build -t deeplasia-service .
docker run -p 8080:8080 -v ./models:/app/models deeplasia-service
```

Navigate to <http://localhost:8080/>

#### Limiting CPU usage

To [limit the CPU usage of the docker container](https://docs.docker.com/config/containers/resource_constraints/), add the following flags to the docker run cmd:

```sh
--cpus=<number_of_cpus>
```

Note, that this should match the number of threads specified with environment variable `DEEPLASIA_THREADS`.

e.g.:

```sh
docker run -p 8080:8080 --cpus=2 -e "DEEPLASIA_THREADS=2" -v ./models:/app/models ghcr.io/crescnet/deeplasia-service
```

## API

[![Swagger UI](https://img.shields.io/badge/-Swagger%20UI-%23Clojure?style=flat&logo=swagger&logoColor=white)](https://crescnet.github.io/deeplasia-service/)

Please refer to `deeplasia-api.yml` for an [OpenAPI](https://www.openapis.org/) specification of the API.

### Request

In python the request can be conducted as follows:

```python
import requests

url = "http://localhost:8080/predict"

test_img = "/path/to/xray.png"
files = { "file": open(test_img, "rb") }

data = {
    "sex": "female",  # specify if known, else is predicted
    "use_mask": True  # default is true
}

resp = requests.post(url, files=files, json=data)
resp.json()
```

Gives something like:

```json
{
    "bone_age": 164.9562530517578,
    "sex_predicted": false,
    "used_sex": "female"
}
```

## Predicting Sex

The canonical way would be as described in previous sections, with using the predicted mask and specifying the sex.
If, however, the sex happens to be unknown (or unsure for e.g. errors of inserting the data) the sex can also be predicted.

## Usage of Masks

Skipping the masking by the predicted mask is meant to be a debugging feature, if the results with the mask are not convincing
(e.g. predicting 127 months as age), one could re-conduct bone age prediction without the mask and see if makes a difference.
We might think about storing the masks as a visual control as well as logging features in general in the future.

## License

The code in this repository and the image `deeplasia-service` are licensed under CC BY-NC-SA 4.0 DEED.
