#!/bin/bash

PHYS_DIR="/home/belletti/Repo/STE-model-calling"

docker run \
    -v "$PHYS_DIR":/ \
    --rm \
    --gpus "device=$CUDA_VISIBLE_DEVICES" \
    ste-llamarecipes-image \
    "/llama-recipes/inference/inference_chat.py" \
    --model_name mistralai/Mistral-7B-v0.1 \
    --data_path ft_datasets/tool_test.json \
    --save_path results/output.json \
    --item_type query \
    --quantization \
    --do_sample \
    --temperature 0.7 \
    --model_type mistral

