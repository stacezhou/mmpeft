ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
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

# config ssh
RUN apt update && apt install -y openssh-server && mkdir /var/run/sshd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN echo $'\
Port 22 \n\
PasswordAuthentication yes \n\
PermitRootLogin yes \n\
' >> /etc/ssh/sshd_config
EXPOSE 22

# set oh-my-zsh
RUN apt update && apt install -y git zsh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN sed -i '1s/bash/zsh/g' /etc/passwd \
    && ZSH=/opt/oh-my-zsh RUNZSH=no CHSH=no \
    sh -c "$(wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"  \
    && mv /root/.zshrc /etc/zsh/zshrc \
    && git clone https://github.com/zsh-users/zsh-syntax-highlighting.git /opt/oh-my-zsh/custom/plugins/zsh-syntax-highlighting \
    && git clone https://github.com/zsh-users/zsh-autosuggestions /opt/oh-my-zsh/custom/plugins/zsh-autosuggestions \
    && sed -i 's/ZSH_THEME=.*/ZSH_THEME="ys"/g' /etc/zsh/zshrc \
    && sed -i 's/plugins=.*/plugins=(git zsh-autosuggestions zsh-syntax-highlighting z)/g' /etc/zsh/zshrc
ENV ZSH_AUTOSUGGEST_HIGHLIGHT_STYLE='fg=243'
RUN mv /etc/zsh/zshrc /root/.zshrc \
    && conda init zsh \
    && mv /root/.zshrc /etc/zsh/zshrc

# config tmux 
RUN apt update && apt install -y tmux
RUN echo $'\
set -g mouse on \n\
set -g default-terminal "xterm-256color" \n\
set -g allow-rename off \n\
set -g mode-keys vi \n\
set -g status-right "" \n\
set -g history-limit 204800 \n\
' >> /etc/tmux.conf
ENV TERM=xterm-256color

# config vim
RUN apt update && apt install -y vim
RUN echo $'\
syntax on \n\
set cindent \n\
set nocompatible \n\
filetype on \n\
set tabstop=4 \n\
set ignorecase \n\
set hisearch \n\
set shiftwidth=4 \n\
set backspace=2 \n\
set number \n\
' >> /etc/vimrc
RUN apt update && apt install -y zip htop

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
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmpretrain.git
WORKDIR ./mmpretrain
RUN mim install --no-cache-dir -e .
RUN pip install ipdb ipython jupyterlab matplotlib pandas tqdm scikit-learn seaborn