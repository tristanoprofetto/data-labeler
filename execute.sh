#!/bin/bash

export OPENAI_API_KEY="YOUR_API_KEY"
source vars.sh

# Function to export variables
export_vars() {
    for key in "${!1}"; do
        export "$key=${!1[$key]}"
    done
}

if [ "$1" == "gpt-labeler" ]; then
    echo "Running GPT Labeler"
    python3 ./gpt/chat.py \ 
        --model_name=$MODEL_NAME \
        --system_prompt_path=$SYSTEM_PROMPT_PATH \
        --input_file_path=$INPUT_FILE_PATH \
        --output_file_path=$OUTPUT_FILE_PATH
elif [ "$1" == "pretrained-labeler" ]; then
    echo "Running Pretrained Labeler"
    export_vars pretrained_vars[@]
    echo "model name: "$PRETRAINED_MODEL_NAME
    echo $PRETRAINED_TOKENIZER_NAME
    echo "class labels: "$PRETRAINED_CLASS_LABELS

    # python3 ./pretrained/chat.py \
    #     --model_name=$MODEL_NAME \
    #     --system_prompt_path=$SYSTEM_PROMPT_PATH \
    #     --input_file_path=$INPUT_FILE_PATH \
    #     --output_file_path=$OUTPUT_FILE_PATH
else
    echo "Invalid argument... must be either gpt-labeler or pretrained-labeler"
    exit 1
fi
