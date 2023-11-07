# Flask setup

## Run in conda environment

```bash
conda create -n flask_ba python=3.9
conda activate flask_ba
pip install -r requirements.txt
python app.py
```

## Docker

To run the application in [docker](https://www.section.io/engineering-education/how-to-deploy-streamlit-app-with-docker/)
use the following command:

```bash
sudo docker build -t flask_bone_age .
sudo docker run -p 8080:8080 -v ./models:/app/models deeplasia-service
```

### Limiting CPU usage

To [limit the CPU usage of the docker container](https://docs.docker.com/config/containers/resource_constraints/), add the following flags to the docker run cmd:

```bash
--cpus=<number_of_cpus>
```

Note, that this should match the number of threads specified by PyTorch (`torch.set_num_threads(threads)` in `app.py`).

# API

## Request

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

So the canonical way would be as described above, with using the predicted mask and specifying the sex.
If, however, the sex happens to be unknown (or unsure for e.g. errors of inserting the data) the sex can also be predicted.
Skipping the masking by the predicted mask is meant to be a debugging feature, if the results with the mask are not convincing (e.g. predicting 127 months as age), one could re-conduct bone age prediction without the mask and see if makes a difference. 
We might think about storing the masks as a visual control as well as logging features in general in the future.