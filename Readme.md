# Streamlit Bone age prediction app


## Data
The [Bone Age models](https://uni-bonn.sciebo.de/apps/files/?dir=/bone2gene%20backup/models/best_models&fileid=1922683453) and [Masking model](https://uni-bonn.sciebo.de/apps/files/?dir=/bone2gene%20backup/masks/models/pretrained_tensormask_cosine/ckp&fileid=1922659654) are available from sciebo (shareable link on request).

## Run

The server can be launched locally using:

```bash
$ streamlit run main.py --server.port=8080
```
The port can be exposed publicly using [ngrok](https://ngrok.com/): `$ ngrok http 8080` (requires registration and setup, see their [website](https://ngrok.com/))

### Arguments
Apart from the [streamlit arguments](https://docs.streamlit.io/library/advanced-features/cli)
the following custom command line arguments are supported:
 * `--n_threads` sets the number of threads PyTorch can use. Set this higher to increase the prediction speed on CPU
 * `--use_cuda` enables running inference using GPU acceleration. Note, that this requires CUDA and, differing from the stated requirements, the CUDA version of PyTorch

Note, that custom arguments need to seperated via `--`, e.g. `$ run main.py --server.port=8080 -- --use_cuda`



## Docker

To run the application in [docker](https://www.section.io/engineering-education/how-to-deploy-streamlit-app-with-docker/)
use the following command:

```bash
$ docker build -t bone_age_streamlit:latest .
$ docker run -p 8080:8080 streamlitapp:latest
```
