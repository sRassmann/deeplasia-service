# Streamlit Bone age prediction app


## Data
The [Bone Age models](https://uni-bonn.sciebo.de/apps/files/?dir=/bone2gene%20backup/models/best_models&fileid=1922683453) and [Masking model](https://uni-bonn.sciebo.de/apps/files/?dir=/bone2gene%20backup/masks/models/pretrained_tensormask_cosine/ckp&fileid=1922659654) are available from sciebo (shareable link on request).

## Run

The server can be launched locally using:

```bash
$ run main.py --server.port=8080
```


or using [docker](https://www.section.io/engineering-education/how-to-deploy-streamlit-app-with-docker/:

```bash
$ docker build -t bone_age_streamlit:latest .
$ docker run -p 8080:8080 streamlitapp:latest
```

The port can be exposed publicly using [ngrok](https://ngrok.com/): `$ ngrok http 8080` (requires registration and setup, see their [website](https://ngrok.com/))
