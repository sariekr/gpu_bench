#!/bin/bash
# --- AYARLAR ---
MODEL_NAME="openai/gpt-oss-20b"
DATASET_PATH="/workspace/data/ShareGPT_V3_unfiltered_cleaned_split.json"
OUTPUT_DIR="/workspace/results/benchmark_$(date +%F_%H-%M-%S)_${MODEL_NAME//\//_}"
GPU_CONFIGS=(1) # Sadece 1 GPU ile test ediyoruz
NUM_PROMPTS=100 # Anlamlı bir test için 100 prompt
GPU_BRAND="nvidia" 

mkdir -p $OUTPUT_DIR
chmod +x ./monitor_gpu.sh

echo "### Starting Benchmark Run ###"
echo "Model: $MODEL_NAME"
echo "GPU Configs: ${GPU_CONFIGS[@]}"
echo "Num Prompts: $NUM_PROMPTS"
echo "Results will be saved in: $OUTPUT_DIR"

for GPUS in "${GPU_CONFIGS[@]}"
do
  echo ""
  echo "--- Running for $GPUS GPU(s) ---"
  MONITOR_LOG_FILE="$OUTPUT_DIR/gpu_usage_${GPUS}_gpus.csv"
  ./monitor_gpu.sh $MONITOR_LOG_FILE $GPU_BRAND &
  MONITOR_PID=$!
  sleep 2
  
  RESULT_FILE="$OUTPUT_DIR/throughput_${GPUS}_gpus.txt"
  
  vllm bench throughput \
    --model $MODEL_NAME \
    --dataset-name sharegpt \
    --dataset-path $DATASET_PATH \
    --tensor-parallel-size $GPUS \
    --num-prompts $NUM_PROMPTS \
    --gpu-memory-utilization 0.8 > $RESULT_FILE
  
  kill $MONITOR_PID
  echo "--- Benchmark completed for $GPUS GPU(s) ---"
done

echo ""
echo "### Benchmark Run Finished! ###"
echo "To see the main result, run:"
echo "cat $OUTPUT_DIR/throughput_1_gpus.txt"