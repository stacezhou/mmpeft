ARG PYTORCH="1.9.1"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
#############################
SHELL [ "/bin/bash","-c" ]
ENV DEBIAN_FRONTEND=noninteractive
# change ubuntu mirror
RUN echo $'\
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse \n\
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse \n\
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse \n\
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse \n\
    '> /etc/apt/sources.list && rm -r /etc/apt/sources.list.d

RUN mkdir /etc/conda && echo $'\
channels: \n\
  - defaults \n\
show_channel_urls: true \n\
default_channels: \n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2 \n\
custom_channels: \n\
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud \n\
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud \n\
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud \n\
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud \n\
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud \n\
    '> /etc/conda/condarc

RUN echo $'\
[global] \n\
index-url = https://pypi.tuna.tsinghua.edu.cn/simple \n\
    '> /etc/pip.conf

RUN apt-get update && apt-get install -y gcc ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1
#########################################

# fetch the key refer to https://forums.developer.nvidia.com/t/18-04-cuda-docker-image-is-broken/212892/9
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 32
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ENV TORCH_CUDA_ARCH_LIST="5.2 6.1 7.0 8.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="(dirname(which conda))/../"

# Install MIM
RUN pip install openmim

# Install MMPretrain
WORKDIR /workspace
RUN git clone https://github.com/stacezhou/mmpeft.git
WORKDIR /workspace/mmpeft
RUN mim install --no-cache-dir -e .
RUN pip install ipdb ipython jupyterlab matplotlib pandas tqdm scikit-learn seaborn ftfy regex wandb 
RUN pip install nni==2.10
RUN pip install "typeguard<3"
