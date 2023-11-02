# Flask setup

## Run in conda environment

```bash
$ conda create -n flask_ba python=3.9
$ conda activate flask_ba
$ pip install -r requirements.txt
$ python app.py
```

## Docker

To run the application in [docker](https://www.section.io/engineering-education/how-to-deploy-streamlit-app-with-docker/)
use the following command:

```bash
$ sudo docker build -t flask_bone_age:latest .
$ sudo docker run -p 8080:8080 flask_bone_age:latest
```

### Limiting CPU usage

To [limit the CPU usage of the docker container](https://docs.docker.com/config/containers/resource_constraints/), add the following flags to the docker run cmd:

```bash
--cpus=<number_of_cpus>
```

Note, that this should match the number of threads specified by PyTorch (`torch.set_num_threads(threads)` in `app.py`).

