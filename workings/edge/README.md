# Inference on an edge device

### Step 1: Clone this GitHub repository

On the edge device, clone this GitHub repository. For example, under your ubuntu edge device:

```bash
cd ~
git clone git@github.com:mon203/w251_final_project.git
```

### Step 2: Setup Conda environment
Setup the environment and install basic requirements using conda. Tensorflow 2.9.1 is compatible with CUDA 11.2 and cuDNN 8.1.

```bash
cd ~/w251_final_project/edge

conda env create -f environment.yml

conda activate casapose
```

### Step 3: Download pretrained weight file (can skip. .h5 file already included)
```bash
# install gdown if you haven't
pip install gdown

cd ~/w251_final_project/edge/data/pretrained_models

# download the pre-train weight file
gdown --id 1uhQT3xgV3c8A83sl4wEwbcL7A0C_5VFx
```

### Step 4: Install a Docker container

To setup and run a CASAPose docker container:

```bash
# build the docker container
docker build -t "casapose:Dockerfile" .

export DATAPATH=~/w251_final_project/edge/import_data/

cd ~/w251_final_project/edge

# run this for a single gpu
docker run -it --net=host --gpus '"device=0"' --ulimit memlock=-1 --ulimit stack=67108864 \
    --rm --shm-size=2g -e "CUDA_VISIBLE_DEVICES=0" -e "NVIDIA_VISIBLE_DEVICES=0" \
    -v $DATAPATH:/workspace/data -v $(pwd):/workspace/CASAPose casapose:Dockerfile bash

# or, run this for multi gpu
docker run -it --net=host --gpus all --ulimit memlock=-1 --ulimit stack=67108864 \
    --rm --shm-size=2g -v $DATAPATH:/workspace/data \
    -v $(pwd):/workspace/CASAPose casapose:Dockerfile bash
```

Once inside the CASAPose docker container, run the following to start the inference.

### Run inference

```bash
python inf_casapose.py -c config/config_8_16.ini \
    --load_h5_weights 1 \
    --load_h5_filename ../../../data/pretrained_models/result_w \
    --datatest $DATAPATH/test/test \
    --datameshes $DATAPATH/test/models \
    --train_vectors_with_ground_truth 0 \
    --save_eval_batches 1
```

