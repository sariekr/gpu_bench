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
GPU_CONFIGS=(1 2 4)
NUM_PROMPTS=10000
GPU_MEMORY_UTILIZATION=0.85
DATASET_PATH="/workspace/data/ShareGPT_V3_unfiltered_cleaned_split.json"
OUTPUT_DIR="/workspace/results/benchmark_$(date +%F_%H-%M-%S)_${MODEL_NAME//\//_}"

# --- OPTIMIZASYON PARAMETRELERI ---
declare -A MAX_NUM_SEQS
MAX_NUM_SEQS[1]=512
MAX_NUM_SEQS[2]=384
MAX_NUM_SEQS[4]=256
MAX_NUM_SEQS[8]=128

# --- ENVIRONMENT OPTIMIZATIONS ---
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DIRECT=1
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
    
    OPTIMAL_MAX_SEQ=${MAX_NUM_SEQS[$GPUS]}
    echo "Using max-num-seqs: $OPTIMAL_MAX_SEQ for $GPUS GPUs"
    
    MONITOR_LOG_FILE="$OUTPUT_DIR/gpu_usage_${GPUS}_gpus.csv"
    RAW_RESULT_FILE="$OUTPUT_DIR/throughput_raw_${GPUS}_gpus.txt"
    
    ./monitor_gpu.sh "$MONITOR_LOG_FILE" "$GPU_BRAND" &
    MONITOR_PID=$!
    sleep 2
    
    if [ "$GPU_BRAND" = "nvidia" ]; then
        echo "INFO: Running NVIDIA optimized command..."
        
        # Warmup
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
            --seed 42 > "$RAW_RESULT_FILE"
    fi
    
    kill $MONITOR_PID 2>/dev/null
    
    # SONUÇ İŞLEME
    echo "--- Processing results... ---"
    SUMMARY_FILE="$OUTPUT_DIR/summary_results_${GPUS}_gpus.txt"
    
    if [ ! -f "$RAW_RESULT_FILE" ]; then
        echo "ERROR: Raw result file not found!"
        continue
    fi
    
    RESULT_LINE=$(grep "Throughput:" "$RAW_RESULT_FILE")
    if [ -z "$RESULT_LINE" ]; then
        echo "ERROR: Throughput data not found!"
        cat "$RAW_RESULT_FILE" | tail -20
        continue
    fi
    
    REQUESTS_PER_SEC=$(echo "$RESULT_LINE" | awk '{print $2}')
    TOTAL_TOKENS_PER_SEC=$(echo "$RESULT_LINE" | awk '{print $4}')
    THROUGHPUT_PER_GPU=$(echo "scale=2; $TOTAL_TOKENS_PER_SEC / $GPUS" | bc -l)
    AVG_LATENCY_MS=$(echo "scale=2; 1000 / $REQUESTS_PER_SEC" | bc -l)
    
    {
        echo "========== BENCHMARK SUMMARY =========="
        echo "Date: $(date)"
        echo "Model: $MODEL_NAME"
        echo "GPU Count: $GPUS"
        echo "Max-num-seqs: $OPTIMAL_MAX_SEQ"
        echo ""
        echo "--- Performance Metrics ---"
        echo "Total Throughput: $TOTAL_TOKENS_PER_SEC tokens/s"
        echo "Per-GPU Throughput: $THROUGHPUT_PER_GPU tokens/s"
        echo "Requests per Second: $REQUESTS_PER_SEC"
        echo "Average Latency: $AVG_LATENCY_MS ms"
        echo "======================================="
    } > "$SUMMARY_FILE"
    
    echo "Summary saved to: $SUMMARY_FILE"
    cat "$SUMMARY_FILE"
done

# --- 2x2 Configuration Test ---
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
    
    # Sonuçları göster
    echo "GPU 0-1 Results:"
    grep "Throughput:" "$OUTPUT_DIR/hybrid_2x2_gpu01.txt" || echo "No results found"
    echo "GPU 2-3 Results:"
    grep "Throughput:" "$OUTPUT_DIR/hybrid_2x2_gpu23.txt" || echo "No results found"
fi

echo ""
echo "### All Benchmarks Completed! ###"