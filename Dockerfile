# see below how to build this docker file (setting huggingface token for llama download)
FROM nvidia/cuda:12.3.2-devel-ubuntu20.04
LABEL maintainer="disi-unibo-nlp"

# Zero interaction (default answers to all questions)
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /home/belletti/STE-model-calling

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

# Pre-download the LLaMA 2 model & tokenizer using authentication
# Use Docker BuildKit secret for Hugging Face authentication
# Before this, I ran on bash:
    #export DOCKER_BUILDKIT=1
    #echo "hf_xxx_your_token_here" > hf_token.txt
# When building:
    #docker build --secret id=hf_token,src=hf_token.txt -t ste-main .
# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install wrapt gdown matplotlib

#AAAAAAAAAAAAAAAA if caching in main works, I won't need this
# # Install Hugging Face dependencies
# RUN pip3 install --no-cache-dir transformers torch torchvision torchaudio accelerate

# # Create a folder to store the Hugging Face model permanently
# RUN mkdir -p /home/belletti/STE-model-calling/huggingface_cache
# # Copy the Hugging Face token secret into the container
# RUN --mount=type=secret,id=hf_token cp /run/secrets/hf_token /home/belletti/STE-model-calling/hf_token.txt
# # Securely pass Hugging Face token using Docker BuildKit secrets
# # Write the Python script to download the model
# RUN echo 'import os' > /home/belletti/STE-model-calling/download_model.py && \
#     echo 'from pathlib import Path' >> /home/belletti/STE-model-calling/download_model.py && \
#     echo 'from transformers import AutoModelForCausalLM, AutoTokenizer' >> /home/belletti/STE-model-calling/download_model.py && \
#     echo 'model_name = "meta-llama/Llama-2-7b-hf"' >> /home/belletti/STE-model-calling/download_model.py && \
#     echo 'token_path = Path("/home/belletti/STE-model-calling/hf_token.txt")' >> /home/belletti/STE-model-calling/download_model.py && \
#     echo 'if not token_path.exists():' >> /home/belletti/STE-model-calling/download_model.py && \
#     echo '    raise ValueError("Hugging Face token is missing. Provide it as a Docker BuildKit secret.")' >> /home/belletti/STE-model-calling/download_model.py && \
#     echo 'token = token_path.read_text().strip()' >> /home/belletti/STE-model-calling/download_model.py && \
#     echo 'AutoTokenizer.from_pretrained(model_name, cache_dir="/home/belletti/STE-model-calling/huggingface_cache", token=token)' >> /home/belletti/STE-model-calling/download_model.py && \
#     echo 'AutoModelForCausalLM.from_pretrained(model_name, cache_dir="/home/belletti/STE-model-calling/huggingface_cache", use_auth_token=token)' >> /home/belletti/STE-model-calling/download_model.py

# # Run the model download script
# RUN python3 /home/belletti/STE-model-calling/download_model.py
# # LOCALLY INSTALL LLAMA
#AAAAAAAAAAAAAAAA if caching in main works, I won't need this

# Ensure Python 3.9 is the default version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --set python3 /usr/bin/python3.9

# Upgrade pip directly without using ensurepip
RUN pip3 install --upgrade pip
RUN pip3 install wrapt --upgrade --ignore-installed
RUN pip3 install gdown
RUN pip3 install matplotlib

# Copy the requirements file into the container
COPY STE/requirements.txt .

# Install dependencies from requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# LOCALLY INSTALL LLAMA
# Set Hugging Face cache directory (optional, but recommended)
ENV HF_HOME="/root/.cache/huggingface"

# Install dependencies
RUN pip install --no-cache-dir transformers torch accelerate

# Back to default frontend
ENV DEBIAN_FRONTEND=dialog