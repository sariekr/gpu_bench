#!/bin/bash

# --- GPU BRAND AUTO-DETECTION ---
if command -v nvidia-smi &> /dev/null; then
    GPU_BRAND="nvidia"
elif command -v rocm-smi &> /dev/null; then
    GPU_BRAND="amd"
else
    echo "ERROR: No supported GPU driver command found."
    exit 1
fi

echo "INFO: Detected GPU Brand: $GPU_BRAND"

# --- AYARLAR ---
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
GPU_CONFIGS=(1 2 4)  # 8 ekleyebilirsiniz
NUM_PROMPTS=10000  # 50000'den düşürdük
GPU_MEMORY_UTILIZATION=0.85  # 0.9'dan düşürdük
DATASET_PATH="/workspace/data/ShareGPT_V3_unfiltered_cleaned_split.json"
OUTPUT_DIR="/workspace/results/benchmark_$(date +%F_%H-%M-%S)_${MODEL_NAME//\//_}"

# --- OPTIMIZASYON PARAMETRELERI ---
# Her GPU sayısı için optimal max-num-seqs
declare -A MAX_NUM_SEQS
MAX_NUM_SEQS[1]=512
MAX_NUM_SEQS[2]=384
MAX_NUM_SEQS[4]=256
MAX_NUM_SEQS[8]=128

# --- ENVIRONMENT OPTIMIZATIONS ---
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DIRECT=1
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_DEDUP_LOGS=0

# --- BENCHMARK BAŞLANGICI ---
mkdir -p "$OUTPUT_DIR"
chmod +x ./monitor_gpu.sh

echo "### Starting Optimized Benchmark Run ###"
echo "Model: $MODEL_NAME"
echo "GPU Configs: ${GPU_CONFIGS[@]}"
echo "Num Prompts: $NUM_PROMPTS"
echo "Results will be saved in: $OUTPUT_DIR"

for GPUS in "${GPU_CONFIGS[@]}"
do
    echo ""
    echo "--- Running Benchmark for $GPUS GPU(s) on $GPU_BRAND platform ---"
    
    # GPU sayısına göre optimal max-num-seqs
    OPTIMAL_MAX_SEQ=${MAX_NUM_SEQS[$GPUS]}
    echo "Using max-num-seqs: $OPTIMAL_MAX_SEQ for $GPUS GPUs"
    
    # Monitoring setup
    MONITOR_LOG_FILE="$OUTPUT_DIR/gpu_usage_${GPUS}_gpus.csv"
    RAW_RESULT_FILE="$OUTPUT_DIR/throughput_raw_${GPUS}_gpus.txt"
    
    ./monitor_gpu.sh "$MONITOR_LOG_FILE" "$GPU_BRAND" &
    MONITOR_PID=$!
    sleep 2
    
    # Platform specific commands
    if [ "$GPU_BRAND" = "nvidia" ]; then
        echo "INFO: Running NVIDIA optimized command..."
        
        # Warmup run (opsiyonel ama önerilen)
        echo "Warming up..."
        vllm bench throughput \
            --model "$MODEL_NAME" \
            --dataset-name sharegpt \
            --dataset-path "$DATASET_PATH" \
            --tensor-parallel-size "$GPUS" \
            --num-prompts 500 \
            --max-num-seqs "$OPTIMAL_MAX_SEQ" \
            --max-model-len 4096 \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" > /dev/null 2>&1
        
        # Actual benchmark
        echo "Running actual benchmark..."
        vllm bench throughput \
            --model "$MODEL_NAME" \
            --dataset-name sharegpt \
            --dataset-path "$DATASET_PATH" \
            --tensor-parallel-size "$GPUS" \
            --num-prompts "$NUM_PROMPTS" \
            --max-num-seqs "$OPTIMAL_MAX_SEQ" \
            --max-model-len 4096 \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --num-scheduler-steps 10 \
            --seed 42 > "$RAW_RESULT_FILE"
            
    elif [ "$GPU_BRAND" = "amd" ]; then
        echo "INFO: Running AMD optimized command..."
        # AMD specific environment
        export VLLM_USE_TRITON_FLASH_ATTN=0
        export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1
        
        python3 /app/vllm/benchmarks/benchmark_throughput.py \
            --model "$MODEL_NAME" \
            --dataset-name sharegpt \
            --dataset-path "$DATASET_PATH" \
            --tensor-parallel-size "$GPUS" \
            --num-prompts "$NUM_PROMPTS" \
            --max-num-seqs "$OPTIMAL_MAX_SEQ" \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --dtype float16 \
            --max-model-len 4096 > "$RAW_RESULT_FILE"
    fi
    
    kill $MONITOR_PID
    echo "--- Benchmark completed. Processing results... ---"
    
    # [Sonuç işleme kısmı aynı kalabilir]
    SUMMARY_FILE="$OUTPUT_DIR/summary_results_${GPUS}_gpus.txt"
    # ... (mevcut processing kodunuz)
done

# --- BONUS: 2x2 Configuration Test ---
if [ ${#GPU_CONFIGS[@]} -gt 0 ] && [ "$GPU_BRAND" = "nvidia" ]; then
    echo ""
    echo "### Testing 2x2 Hybrid Configuration ###"
    
    # Instance 1: GPU 0,1
    CUDA_VISIBLE_DEVICES=0,1 vllm bench throughput \
        --model "$MODEL_NAME" \
        --dataset-name sharegpt \
        --dataset-path "$DATASET_PATH" \
        --tensor-parallel-size 2 \
        --num-prompts 5000 \
        --max-num-seqs 256 \
        --max-model-len 4096 > "$OUTPUT_DIR/hybrid_2x2_gpu01.txt" &
    PID1=$!
    
    # Instance 2: GPU 2,3
    CUDA_VISIBLE_DEVICES=2,3 vllm bench throughput \
        --model "$MODEL_NAME" \
        --dataset-name sharegpt \
        --dataset-path "$DATASET_PATH" \
        --tensor-parallel-size 2 \
        --num-prompts 5000 \
        --max-num-seqs 256 \
        --max-model-len 4096 > "$OUTPUT_DIR/hybrid_2x2_gpu23.txt" &
    PID2=$!
    
    # Wait for both
    wait $PID1 $PID2
    
    echo "2x2 Hybrid configuration test completed"
fi

echo ""
echo "### All Benchmarks Completed! ###"