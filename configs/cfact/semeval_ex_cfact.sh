#!/bin/bash

set -e

BASE_DIR=/shared/aifiles/disk1/media/artifacts/LLaVA-Med/VQA-RAD
DATA_PATH=/users/max/data/corpora/semeval-2016
PRED_PATH=/shared/aifiles/disk1/media/artifacts/cfact/semeval-2016

mkdir -p $PRED_PATH

# cfact GPT-4, GPT-3.5
declare -a configs=("gpt4-target" "gpt3.5-target")

for config in "${configs[@]}"
do
    echo "Running $config"

    python gpt_gleam/unlabeled.py \
        --config configs/cfact/cfact-$config.yaml \
        --data_path $DATA_PATH/test.jsonl \
        --frame_path $DATA_PATH/frames.json \
        --output_path $PRED_PATH/cfact-$config.jsonl

    python gpt_gleam/verify.py \
        --config configs/cfact/verify-$config.yaml \
        --frame_path $DATA_PATH/frames.json \
        --data_path $DATA_PATH/test.jsonl \
        --cfact_path $PRED_PATH/cfact-$config.jsonl \
        --output_path $PRED_PATH/verify-$config.jsonl

    python gpt_gleam/eval.py \
        --data_path $DATA_PATH/test.jsonl \
        --frame_path $DATA_PATH/frames.json \
        --pred_path $PRED_PATH/verify-$config.jsonl \
        --output_path $PRED_PATH/results-cfact-$config-test.txt
done
