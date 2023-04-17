FROM nvcr.io/nvidia/tensorflow:22.06-tf2-py3

RUN apt -y update && apt -y install libsm6 libxrender1 libfontconfig1 libxtst6 libxi6
RUN apt -y install libgl1 libglfw3-dev libgles2-mesa-dev
RUN apt autoremove && apt clean

RUN pip install --upgrade pip
RUN pip install tensorflow-addons==0.17.0 \
        opencv-python==4.5.5.62 \
        numpy==1.23.1

ENV WORKPATH=/workspace/CASAPose/
ENV DATAPATH=/workspace/data/

WORKDIR $WORKPATH
CMD ["bash"]