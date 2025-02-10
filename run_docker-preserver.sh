#!/bin/bash

# Define environment variables
HUGGING_FACE_HUB_TOKEN="<<YOUR_HUGGING_FACE_HUB_TOKEN>>"
HF_HOME="/huggingface_cache"
MODEL_CKPT="meta-llama/Meta-Llama-3-8B-Instruct"
NUM_EPISODES=3
NUM_STM_SLOTS=3
MAX_TURN=3
DIR_WRITE="STE/results/"
RAPIDAPI_KEY="YOUR_RAPIDAPI_KEY"

# Run the Docker container
docker run \
  -v "/home/belletti/STE-model-calling/STE:/STE" \
  -v "/home/belletti/huggingface_cache:/huggingface_cache" \
  -v "/home/belletti/STE-model-calling/STE/results:/STE/results" \
  --rm \
  --gpus "device=$CUDA_VISIBLE_DEVICES" \
  -e HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN" \
  -e HF_HOME="$HF_HOME" \
  ste-main \
  /bin/bash -c "
    pip install accelerate==0.26.0 evaluate nltk absl-py rouge-score sacrebleu torch transformers numpy scipy scikit-learn bert_score &&
    python3 /STE/main.py \
      --model_ckpt $MODEL_CKPT \
      --num_episodes $NUM_EPISODES \
      --num_stm_slots $NUM_STM_SLOTS \
      --max_turn $MAX_TURN \
      --dir_write $DIR_WRITE \
      --rapidapi_key $RAPIDAPI_KEY \
      --if_visualize
    "