#!/bin/bash

set -e

BASE_DIR=/shared/aifiles/disk1/media/artifacts/LLaVA-Med/VQA-RAD
DATA_PATH=/users/max/data/corpora/semeval-2016
PRED_PATH=/shared/aifiles/disk1/media/artifacts/cfact/semeval-2016

mkdir -p $PRED_PATH

# direct GPT-4, direct GPT-3.5, CoT GPT-4, CoT GPT-3.5
declare -a configs=("direct-gpt4" "direct-gpt3.5" "cot-gpt4" "cot-gpt3.5")

for config in "${configs[@]}"
do
    echo "Running $config"

    python gpt_gleam/labeled.py \
    --config configs/cfact/$config.yaml \
    --data_path $DATA_PATH/test.jsonl \
    --frame_path $DATA_PATH/frames.json \
    --output_path $PRED_PATH/$config.jsonl

    python gpt_gleam/eval.py \
        --data_path $DATA_PATH/test.jsonl \
        --frame_path $DATA_PATH/frames.json \
        --pred_path $PRED_PATH/$config.jsonl \
        --output_path $PRED_PATH/results-$config-test.txt
done
