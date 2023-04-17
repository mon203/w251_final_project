# EC2 Edge Implementation

### Step 1: Set up an AWS EC2 instance

The project used a g5.xlarge instance with an Ubuntu 20.04 - Deep Learning AMI (GPU CUDA 11.2.1).

### Step 2: Clone the GitHub repository

On the edge device, clone this GitHub repository. For example, under your ubuntu edge device:

```bash
cd ~
git clone git@github.com:mon203/w251_final_project.git
```

### Step 3: Build and run the Docker container

To setup and run a CASAPose Docker container:

```bash
# build the docker container
docker build -t "casapose:edge" .

export DATAPATH=~/w251_final_project/edge_final/import_data/

cd ~/w251_final_project/edge_final

# Docker run command for a single GPU
docker run -it --net=host --gpus '"device=0"' --ulimit memlock=-1 --ulimit stack=67108864 \
    --rm --shm-size=2g -e "CUDA_VISIBLE_DEVICES=0" -e "NVIDIA_VISIBLE_DEVICES=0" \
    -v $DATAPATH:/workspace/data -v $(pwd):/workspace/CASAPose casapose:edge bash

# Docker run command for multi GPUs
docker run -it --net=host --gpus all --ulimit memlock=-1 --ulimit stack=67108864 \
    --rm --shm-size=2g -v $DATAPATH:/workspace/data \
    -v $(pwd):/workspace/CASAPose casapose:edge bash
```

Once inside the CASAPose docker container, run the following to start the inference.

### Step 4: Run inference

To run the inference .py files:
```bash
# For frames
python3 final_inference_frames.py

# For videos
python3 final_inference_video.py
```

Videos should be preprocessed using the [video_processor.ipynb](video_processor.ipynb).


To view inference step by step, JupyterLab must be installed:
```bash
pip3 install jupyterlab

jupyter lab --allow-root -ip=0.0.0.0
```

Open [final_inf_frames.ipynb](final_inf_frames.ipynb) or [final_inf_video.ipynb](final_inf_video.ipynb).
