# # Use NVIDIA CUDA base image with Ubuntu 20.04
# #FROM nvidia/cuda:12.3.2-devel-ubuntu20.04
# FROM huggingface/transformers-pytorch-latest-gpu
# # Set non-interactive mode for apt commands
# ENV DEBIAN_FRONTEND=noninteractive

# # Set work directory
# WORKDIR /

# # Install general-purpose dependencies and Python 3.9
# RUN \
#     apt-get update -y && \
#     apt-get install -y curl \
#     git \
#     bash \
#     nano \
#     wget \
#     software-properties-common && \
#     add-apt-repository ppa:deadsnakes/ppa && \
#     apt-get update -y && \
#     apt-get install -y python3.9 python3.9-distutils python3-pip && \
#     apt-get autoremove -y && \
#     apt-get clean -y && \
#     rm -rf /var/lib/apt/lists/*

# # Ensure Python 3.9 is the default Python version
# RUN ln -sf /usr/bin/python3.9 /usr/bin/python3

# # Upgrade pip
# RUN pip install --upgrade pip

# # Install necessary Python packages
# RUN pip install wrapt --upgrade --ignore-installed && \
#     pip install gdown && \
#     pip install --default-timeout=100 future

# # Copy the requirements file into the container
# COPY llama-recipes/requirements.txt .

# # Install dependencies from requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # Install additional dependencies from GitHub
# RUN pip install git+https://github.com/huggingface/accelerate.git && \
#     pip install git+https://github.com/huggingface/trl.git

# # Install flash-attn and other specific dependencies
# RUN pip install --upgrade packaging ninja && \
#     #pip install flash-attn && \
#     pip install --upgrade flash-attn==2.3.4 --no-build-isolation && \
#     pip install --upgrade bitsandbytes==0.43.0 && \
#     pip install vllm

# # Set back the default frontend mode


# ENV DEBIAN_FRONTEND=dialog

# --------------------------------------------------------------------------------------------

# FROM huggingface/transformers-pytorch-latest-gpu

# # Set work directory
# WORKDIR /

# # Copy the requirements file into the container at /workspace
# COPY llama-recipes/requirements.txt .

# # Install dependencies from requirements.txt
# # Upgrade pip before installing dependencies
# RUN pip3.8 install --upgrade pip && \
#     pip3.8 install --no-cache-dir -r requirements.txt
# --------------------------------------------------------------------------------------------

# FROM nvidia/cuda:12.3.2-devel-ubuntu20.04
# LABEL maintainer="disi-unibo-nlp"

# # Zero interaction (default answers to all questions)
# ENV DEBIAN_FRONTEND=noninteractive

# # Set work directory
# WORKDIR /

# # Install general-purpose dependencies and Python 3.9
# RUN apt-get update -y && \
#     apt-get install -y curl \
#     git \
#     bash \
#     nano \
#     wget \
#     python3.9 \
#     python3.9-distutils \
#     python3-pip && \
#     apt-get autoremove -y && \
#     apt-get clean -y && \
#     rm -rf /var/lib/apt/lists/*

# # Ensure Python 3.9 is the default version
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
#     update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.9 1

# # Upgrade pip
# RUN pip3 install --upgrade pip
# RUN pip3 install wrapt --upgrade --ignore-installed
# RUN pip3 install gdown

# # Copy the requirements file into the container
# COPY llama-recipes/requirements.txt .

# # Install dependencies from requirements.txt
# RUN pip3 install --no-cache-dir -r requirements.txt

# # Back to default frontend
# ENV DEBIAN_FRONTEND=dialog

FROM nvidia/cuda:12.3.2-devel-ubuntu20.04
LABEL maintainer="disi-unibo-nlp"

# Zero interaction (default answers to all questions)
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /

# Install general-purpose dependencies and Python 3.9
RUN apt-get update -y && \
    apt-get install -y curl \
    git \
    bash \
    nano \
    wget \
    python3.9 \
    python3.9-distutils \
    python3.9-venv \
    python3-pip && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Ensure Python 3.9 is the default version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --set python3 /usr/bin/python3.9

# Upgrade pip directly without using ensurepip
RUN pip3 install --upgrade pip
RUN pip3 install wrapt --upgrade --ignore-installed
RUN pip3 install gdown

# Copy the requirements file into the container
COPY llama-recipes/requirements.txt .

# Install dependencies from requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Back to default frontend
ENV DEBIAN_FRONTEND=dialog