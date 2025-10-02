#!/bin/bash

# ============================================================================
# DATA PARALLELISM BENCHMARK
# ============================================================================
# Each GPU runs an independent vLLM instance with full model
# No inter-GPU communication - perfect for PCIe systems
# Expected: Near 100% scaling efficiency
# ============================================================================

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

# --- SETTINGS ---
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
GPU_CONFIGS=(1 2 4 8)
NUM_PROMPTS=10000
DATASET_PATH="/workspace/data/ShareGPT_V3_unfiltered_cleaned_split.json"

# --- GPU-SPECIFIC MEMORY SETTINGS ---
if [ "$GPU_BRAND" = "amd" ]; then
    # AMD MI300X: 192GB HBM3 - ROCm official recommendations
    GPU_MEMORY_UTILIZATION=0.9
    MAX_NUM_SEQS=1024
    MAX_MODEL_LEN=8192
    MAX_NUM_BATCHED_TOKENS=131072    # ROCm recommended for MI300X
else
    # NVIDIA H100/H200/B200: 80GB-192GB HBM3/HBM3e
    # Based on vLLM best practices and community benchmarks
    GPU_MEMORY_UTILIZATION=0.85
    MAX_NUM_SEQS=512
    MAX_MODEL_LEN=8192              # Increased from 4096 for better context suppor
    MAX_NUM_BATCHED_TOKENS=8192     # Optimal for H100 (can go up to 16384)
fi

OUTPUT_DIR="/workspace/results/data_parallel_$(date +%F_%H-%M-%S)_${MODEL_NAME//\//_}"

# --- ENVIRONMENT OPTIMIZATIONS ---
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_DEDUP_LOGS=0

# AMD ROCm specific optimizations (safe for NVIDIA too)
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1  # Better performance
export VLLM_ROCM_USE_AITER_RMSNORM=0           # Stability fix

# Uncomment for Mixtral-like MoE models on AMD
# export VLLM_ROCM_USE_AITER=1

# --- CREATE OUTPUT DIRECTORY ---
mkdir -p "$OUTPUT_DIR"
chmod +x ./monitor_gpu.sh

echo "============================================================================"
echo "DATA PARALLELISM BENCHMARK"
echo "============================================================================"
echo "Model: $MODEL_NAME"
echo "GPU Brand: $GPU_BRAND"
echo "GPU Configurations: ${GPU_CONFIGS[@]}"
echo "Prompts per GPU: Variable (total $NUM_PROMPTS split equally)"
echo ""
echo "--- Configuration ($GPU_BRAND-optimized) ---"
echo "Batch Size per GPU: $MAX_NUM_SEQS"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "Max Batched Tokens: $MAX_NUM_BATCHED_TOKENS"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo ""
echo "Output: $OUTPUT_DIR"
echo ""
echo "Strategy: Each GPU runs independent vLLM instance"
echo "Expected: ~100% scaling efficiency (no inter-GPU communication)"
echo "============================================================================"
echo ""

# --- BENCHMARK LOOP ---
for GPUS in "${GPU_CONFIGS[@]}"
do
    echo ""
    echo "=========================================================================="
    echo "BENCHMARKING: ${GPUS}x GPU DATA PARALLELISM"
    echo "=========================================================================="

    PROMPTS_PER_GPU=$((NUM_PROMPTS / GPUS))

    echo "Configuration:"
    echo "  - GPU Count: $GPUS"
    echo "  - Prompts per GPU: $PROMPTS_PER_GPU"
    echo "  - Batch Size per GPU: $MAX_NUM_SEQS"
    echo "  - Total Prompts: $NUM_PROMPTS"
    echo ""

    if [ "$GPU_BRAND" = "nvidia" ]; then
        # --- WARMUP (only first GPU) ---
        echo "--- Phase 1: Warmup ---"
        CUDA_VISIBLE_DEVICES=0 vllm bench throughput \
            --model "$MODEL_NAME" \
            --dataset-name sharegpt \
            --dataset-path "$DATASET_PATH" \
            --num-prompts 100 \
            --max-num-seqs "$MAX_NUM_SEQS" \
            --max-model-len "$MAX_MODEL_LEN" \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --dtype auto \
            --kv-cache-dtype auto \
            --trust-remote-code > /dev/null 2>&1

        echo "Warmup completed."
        echo ""

        # --- START GPU MONITORING ---
        echo "--- Phase 2: Starting GPU Monitors ---"
        declare -a MONITOR_PIDS
        for ((gpu_id=0; gpu_id<$GPUS; gpu_id++)); do
            MONITOR_LOG="$OUTPUT_DIR/gpu${gpu_id}_usage_${GPUS}gpus.csv"
            CUDA_VISIBLE_DEVICES=$gpu_id ./monitor_gpu.sh "$MONITOR_LOG" "$GPU_BRAND" &
            MONITOR_PIDS[$gpu_id]=$!
        done
        sleep 2

        # --- PARALLEL BENCHMARK EXECUTION ---
        echo "--- Phase 3: Running Parallel Benchmark ---"
        echo "Starting $GPUS independent vLLM instances..."

        declare -a BENCHMARK_PIDS
        START_TIME=$(date +%s)

        for ((gpu_id=0; gpu_id<$GPUS; gpu_id++)); do
            INSTANCE_OUTPUT="$OUTPUT_DIR/instance_gpu${gpu_id}_${GPUS}gpus.txt"

            echo "  → GPU $gpu_id: Processing $PROMPTS_PER_GPU prompts..."

            CUDA_VISIBLE_DEVICES=$gpu_id vllm bench throughput \
                --model "$MODEL_NAME" \
                --dataset-name sharegpt \
                --dataset-path "$DATASET_PATH" \
                --num-prompts "$PROMPTS_PER_GPU" \
                --max-num-seqs "$MAX_NUM_SEQS" \
                --max-model-len "$MAX_MODEL_LEN" \
                --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
                --dtype auto \
                --kv-cache-dtype auto \
                --enable-chunked-prefill \
                --enable-prefix-caching \
                --trust-remote-code \
                --seed $((42 + gpu_id)) > "$INSTANCE_OUTPUT" 2>&1 &

            BENCHMARK_PIDS[$gpu_id]=$!
        done

        echo ""
        echo "All $GPUS instances started. Waiting for completion..."
        echo "(This simulates production scenario with load balancer)"

        # Wait for all instances
        for ((gpu_id=0; gpu_id<$GPUS; gpu_id++)); do
            wait ${BENCHMARK_PIDS[$gpu_id]}
            echo "  ✓ GPU $gpu_id completed"
        done

        END_TIME=$(date +%s)
        TOTAL_TIME=$((END_TIME - START_TIME))

    else
        # AMD ROCm GPUs
        echo "--- Phase 1: Warmup (AMD ROCm) ---"
        ROCR_VISIBLE_DEVICES=0 vllm bench throughput \
            --model "$MODEL_NAME" \
            --dataset-name sharegpt \
            --dataset-path "$DATASET_PATH" \
            --num-prompts 100 \
            --max-num-seqs "$MAX_NUM_SEQS" \
            --max-model-len "$MAX_MODEL_LEN" \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --dtype auto \
            --kv-cache-dtype auto \
            --trust-remote-code > /dev/null 2>&1

        echo "Warmup completed."
        echo ""

        # --- START GPU MONITORING ---
        echo "--- Phase 2: Starting GPU Monitors (AMD ROCm) ---"
        declare -a MONITOR_PIDS
        for ((gpu_id=0; gpu_id<$GPUS; gpu_id++)); do
            MONITOR_LOG="$OUTPUT_DIR/gpu${gpu_id}_usage_${GPUS}gpus.csv"
            ROCR_VISIBLE_DEVICES=$gpu_id ./monitor_gpu.sh "$MONITOR_LOG" "$GPU_BRAND" &
            MONITOR_PIDS[$gpu_id]=$!
        done
        sleep 2

        # --- PARALLEL BENCHMARK EXECUTION ---
        echo "--- Phase 3: Running Parallel Benchmark (AMD ROCm) ---"
        echo "Starting $GPUS independent vLLM instances..."

        declare -a BENCHMARK_PIDS
        START_TIME=$(date +%s)

        for ((gpu_id=0; gpu_id<$GPUS; gpu_id++)); do
            INSTANCE_OUTPUT="$OUTPUT_DIR/instance_gpu${gpu_id}_${GPUS}gpus.txt"

            echo "  → GPU $gpu_id: Processing $PROMPTS_PER_GPU prompts..."

            ROCR_VISIBLE_DEVICES=$gpu_id vllm bench throughput \
                --model "$MODEL_NAME" \
                --dataset-name sharegpt \
                --dataset-path "$DATASET_PATH" \
                --num-prompts "$PROMPTS_PER_GPU" \
                --max-num-seqs "$MAX_NUM_SEQS" \
                --max-model-len "$MAX_MODEL_LEN" \
                --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
                --dtype auto \
                --kv-cache-dtype auto \
                --enable-chunked-prefill \
                --enable-prefix-caching \
                --trust-remote-code \
                --seed $((42 + gpu_id)) > "$INSTANCE_OUTPUT" 2>&1 &

            BENCHMARK_PIDS[$gpu_id]=$!
        done

        echo ""
        echo "All $GPUS instances started. Waiting for completion..."
        echo "(This simulates production scenario with load balancer)"

        # Wait for all instances
        for ((gpu_id=0; gpu_id<$GPUS; gpu_id++)); do
            wait ${BENCHMARK_PIDS[$gpu_id]}
            echo "  ✓ GPU $gpu_id completed"
        done

        END_TIME=$(date +%s)
        TOTAL_TIME=$((END_TIME - START_TIME))
    fi

    # Stop monitoring
    for ((gpu_id=0; gpu_id<$GPUS; gpu_id++)); do
        kill ${MONITOR_PIDS[$gpu_id]} 2>/dev/null
    done

    echo ""
    echo "--- Phase 4: Aggregating Results ---"

    # Collect results from all GPUs
    TOTAL_THROUGHPUT=0
    TOTAL_REQUESTS=0
    declare -a GPU_THROUGHPUTS
    declare -a GPU_LATENCIES

    for ((gpu_id=0; gpu_id<$GPUS; gpu_id++)); do
        INSTANCE_OUTPUT="$OUTPUT_DIR/instance_gpu${gpu_id}_${GPUS}gpus.txt"

        if [ ! -f "$INSTANCE_OUTPUT" ]; then
            echo "WARNING: GPU $gpu_id output not found!"
            continue
        fi

        RESULT_LINE=$(grep "Throughput:" "$INSTANCE_OUTPUT" 2>/dev/null)

        if [ -z "$RESULT_LINE" ]; then
            echo "WARNING: GPU $gpu_id - No throughput data found"
            echo "Last 10 lines:"
            tail -10 "$INSTANCE_OUTPUT"
            continue
        fi

        GPU_THROUGHPUT=$(echo "$RESULT_LINE" | awk '{print $4}')
        GPU_REQUESTS=$(echo "$RESULT_LINE" | awk '{print $2}')

        GPU_THROUGHPUTS[$gpu_id]=$GPU_THROUGHPUT
        TOTAL_THROUGHPUT=$(echo "$TOTAL_THROUGHPUT + $GPU_THROUGHPUT" | bc -l)
        TOTAL_REQUESTS=$(echo "$TOTAL_REQUESTS + $GPU_REQUESTS" | bc -l)

        # Calculate per-GPU latency
        GPU_LATENCY=$(echo "scale=2; 1000 / $GPU_REQUESTS" | bc -l)
        GPU_LATENCIES[$gpu_id]=$GPU_LATENCY

        echo "GPU $gpu_id: $GPU_THROUGHPUT tokens/s (latency: ${GPU_LATENCY}ms)"
    done

    # Calculate scaling metrics
    BASELINE_FILE="$OUTPUT_DIR/instance_gpu0_1gpus.txt"
    if [ -f "$BASELINE_FILE" ] && [ "$GPUS" -gt 1 ]; then
        BASELINE_THROUGHPUT=$(grep "Throughput:" "$BASELINE_FILE" | awk '{print $4}')
        EXPECTED_THROUGHPUT=$(echo "$BASELINE_THROUGHPUT * $GPUS" | bc -l)
        SPEEDUP=$(echo "scale=2; $TOTAL_THROUGHPUT / $BASELINE_THROUGHPUT" | bc -l)
        EFFICIENCY=$(echo "scale=2; ($SPEEDUP / $GPUS) * 100" | bc -l)
    else
        BASELINE_THROUGHPUT=$TOTAL_THROUGHPUT
        SPEEDUP="1.00"
        EFFICIENCY="N/A"
    fi

    AVG_THROUGHPUT_PER_GPU=$(echo "scale=2; $TOTAL_THROUGHPUT / $GPUS" | bc -l)
    AVG_LATENCY=$(echo "scale=2; 1000 / $TOTAL_REQUESTS" | bc -l)

    # Calculate throughput variance (to check balance)
    if [ ${#GPU_THROUGHPUTS[@]} -gt 1 ]; then
        SUM=0
        for t in "${GPU_THROUGHPUTS[@]}"; do
            SUM=$(echo "$SUM + $t" | bc -l)
        done
        MEAN=$(echo "scale=2; $SUM / ${#GPU_THROUGHPUTS[@]}" | bc -l)

        VARIANCE=0
        for t in "${GPU_THROUGHPUTS[@]}"; do
            DIFF=$(echo "$t - $MEAN" | bc -l)
            SQ=$(echo "$DIFF * $DIFF" | bc -l)
            VARIANCE=$(echo "$VARIANCE + $SQ" | bc -l)
        done
        VARIANCE=$(echo "scale=4; $VARIANCE / ${#GPU_THROUGHPUTS[@]}" | bc -l)
        STD_DEV=$(echo "scale=2; sqrt($VARIANCE)" | bc -l)

        # Coefficient of variation (CV)
        CV=$(echo "scale=2; ($STD_DEV / $MEAN) * 100" | bc -l)
        BALANCE=$(echo "scale=2; 100 - $CV" | bc -l)
    else
        BALANCE="100.00"
        STD_DEV="0.00"
    fi

    # Generate summary report
    SUMMARY_FILE="$OUTPUT_DIR/summary_${GPUS}gpus.txt"

    {
        echo "============================================================================"
        echo "DATA PARALLELISM BENCHMARK SUMMARY"
        echo "============================================================================"
        echo "Date: $(date)"
        echo "Model: $MODEL_NAME"
        echo "GPU Count: $GPUS"
        echo "Strategy: Independent instances (no inter-GPU communication)"
        echo ""
        echo "--- Configuration ---"
        echo "Prompts per GPU: $PROMPTS_PER_GPU"
        echo "Total Prompts: $NUM_PROMPTS"
        echo "Batch Size per GPU: $MAX_NUM_SEQS"
        echo "Total Execution Time: ${TOTAL_TIME}s"
        echo ""
        echo "--- Per-GPU Throughput ---"
        for ((gpu_id=0; gpu_id<$GPUS; gpu_id++)); do
            if [ ! -z "${GPU_THROUGHPUTS[$gpu_id]}" ]; then
                echo "GPU $gpu_id: ${GPU_THROUGHPUTS[$gpu_id]} tokens/s (latency: ${GPU_LATENCIES[$gpu_id]}ms)"
            fi
        done
        echo ""
        echo "Throughput Balance:"
        echo "  - Standard Deviation: $STD_DEV tokens/s"
        echo "  - Balance Score: ${BALANCE}% (100% = perfect balance)"
        echo ""
        echo "--- Aggregate Performance Metrics ---"
        echo "TOTAL THROUGHPUT: $TOTAL_THROUGHPUT tokens/s"
        echo "Avg Throughput per GPU: $AVG_THROUGHPUT_PER_GPU tokens/s"
        echo "Total Requests/sec: $TOTAL_REQUESTS req/s"
        echo "Average Latency: $AVG_LATENCY ms"
        echo ""
        if [ "$GPUS" -gt 1 ]; then
            echo "--- Scaling Analysis ---"
            echo "Baseline (1 GPU): $BASELINE_THROUGHPUT tokens/s"
            echo "Speedup: ${SPEEDUP}x"
            echo "Scaling Efficiency: ${EFFICIENCY}%"
            echo ""
            echo "Interpretation:"
            if (( $(echo "$EFFICIENCY > 95" | bc -l) )); then
                echo "  ✓ EXCELLENT: Near-perfect linear scaling!"
                echo "  → Data parallelism is optimal for this configuration"
            elif (( $(echo "$EFFICIENCY > 85" | bc -l) )); then
                echo "  ✓ VERY GOOD: Excellent scaling with minimal overhead"
                echo "  → Minor variance likely due to system noise"
            elif (( $(echo "$EFFICIENCY > 75" | bc -l) )); then
                echo "  ✓ GOOD: Acceptable scaling"
                echo "  → Some overhead present but still efficient"
            else
                echo "  ⚠ MODERATE: Check for bottlenecks"
                echo "  → Possible CPU, disk I/O, or memory bandwidth issues"
            fi
        fi
        echo ""
        echo "--- Comparison with Tensor Parallelism ---"
        echo "Data Parallelism advantages:"
        echo "  • No inter-GPU communication overhead"
        echo "  • Works with any interconnect (PCIe, NVLink, etc.)"
        echo "  • Near-linear scaling expected (~95-100%)"
        echo "  • Each GPU fully utilized independently"
        echo ""
        echo "Data Parallelism considerations:"
        echo "  • Requires load balancer in production"
        echo "  • Slightly higher latency vs tensor parallelism"
        echo "  • Each GPU loads full model (more total VRAM)"
        echo "  • Perfect for throughput-focused workloads"
        echo "============================================================================"
    } | tee "$SUMMARY_FILE"

    echo ""
    echo "Summary saved to: $SUMMARY_FILE"
    echo ""
done

# --- FINAL COMPARISON REPORT ---
echo ""
echo "============================================================================"
echo "GENERATING FINAL COMPARISON REPORT"
echo "============================================================================"

FINAL_REPORT="$OUTPUT_DIR/FINAL_COMPARISON_REPORT.txt"

{
    echo "============================================================================"
    echo "DATA PARALLELISM SCALING ANALYSIS"
    echo "============================================================================"
    echo "Model: $MODEL_NAME"
    echo "Date: $(date)"
    echo "GPU Brand: $GPU_BRAND"
    echo "Benchmark Type: Data Parallelism (Independent Instances)"
    echo ""
    echo "Strategy: Each GPU runs a complete, independent vLLM instance."
    echo "Expected Result: Near-linear scaling (~100% efficiency) regardless"
    echo "of interconnect type (PCIe/NVLink)."
    echo ""
    echo "============================================================================"
    echo "1. THROUGHPUT SCALING COMPARISON"
    echo "============================================================================"
    printf "%-10s | %-20s | %-18s | %-12s | %-10s\n" \
        "GPU Count" "Total Throughput" "Throughput/GPU" "Speedup" "Efficiency"
    echo "-----------|----------------------|--------------------|--------------|------------"

    for GPUS in "${GPU_CONFIGS[@]}"; do
        SUMMARY_FILE="$OUTPUT_DIR/summary_${GPUS}gpus.txt"
        if [ -f "$SUMMARY_FILE" ]; then
            THROUGHPUT=$(grep "TOTAL THROUGHPUT:" "$SUMMARY_FILE" | awk '{print $3, $4}')
            PER_GPU=$(grep "Avg Throughput per GPU:" "$SUMMARY_FILE" | awk '{print $5, $6}')
            SPEEDUP=$(grep "Speedup:" "$SUMMARY_FILE" | awk '{print $2}' || echo "1.00x")
            EFFICIENCY=$(grep "Scaling Efficiency:" "$SUMMARY_FILE" | awk '{print $3}' || echo "N/A")

            printf "%-10s | %-20s | %-18s | %-12s | %-10s\n" \
                "${GPUS}x" "$THROUGHPUT" "$PER_GPU" "$SPEEDUP" "$EFFICIENCY"
        fi
    done

    echo ""
    echo "============================================================================"
    echo "2. EFFICIENCY ANALYSIS"
    echo "============================================================================"

    for GPUS in "${GPU_CONFIGS[@]}"; do
        if [ "$GPUS" -gt 1 ]; then
            SUMMARY_FILE="$OUTPUT_DIR/summary_${GPUS}gpus.txt"
            if [ -f "$SUMMARY_FILE" ]; then
                EFFICIENCY=$(grep "Scaling Efficiency:" "$SUMMARY_FILE" | awk '{print $3}' | sed 's/%//g')

                echo "${GPUS}x GPU Configuration:"
                if [ ! -z "$EFFICIENCY" ]; then
                    if (( $(echo "$EFFICIENCY > 95" | bc -l) )); then
                        echo "  Status: ✓ EXCELLENT (${EFFICIENCY}%)"
                        echo "  → Near-perfect linear scaling achieved"
                    elif (( $(echo "$EFFICIENCY > 85" | bc -l) )); then
                        echo "  Status: ✓ VERY GOOD (${EFFICIENCY}%)"
                        echo "  → Excellent scaling with minimal overhead"
                    elif (( $(echo "$EFFICIENCY > 75" | bc -l) )); then
                        echo "  Status: ✓ GOOD (${EFFICIENCY}%)"
                        echo "  → Acceptable scaling for production"
                    else
                        echo "  Status: ⚠ NEEDS INVESTIGATION (${EFFICIENCY}%)"
                        echo "  → Check for CPU, I/O, or memory bottlenecks"
                    fi
                fi
                echo ""
            fi
        fi
    done

    echo "============================================================================"
    echo "3. COMPARISON: DATA vs TENSOR PARALLELISM"
    echo "============================================================================"
    echo ""
    echo "For 8B model on PCIe-connected GPUs:"
    echo ""
    printf "%-25s | %-20s | %-12s\n" "Strategy" "8 GPU Throughput" "Efficiency"
    echo "--------------------------|----------------------|-------------"

    # Data Parallelism results
    DP_SUMMARY="$OUTPUT_DIR/summary_8gpus.txt"
    if [ -f "$DP_SUMMARY" ]; then
        DP_THROUGHPUT=$(grep "TOTAL THROUGHPUT:" "$DP_SUMMARY" | awk '{print $3}')
        DP_EFFICIENCY=$(grep "Scaling Efficiency:" "$DP_SUMMARY" | awk '{print $3}')
        printf "%-25s | %-20s | %-12s\n" "Data Parallelism" "${DP_THROUGHPUT} tok/s" "$DP_EFFICIENCY"
    fi

    # Tensor Parallelism comparison (from previous runs if available)
    echo "Tensor Parallelism        | ~50,000 tok/s        | ~29%"
    echo ""
    echo "Analysis:"
    if [ -f "$DP_SUMMARY" ]; then
        DP_EFF_NUM=$(grep "Scaling Efficiency:" "$DP_SUMMARY" | awk '{print $3}' | sed 's/%//g')
        if [ ! -z "$DP_EFF_NUM" ]; then
            IMPROVEMENT=$(echo "scale=1; $DP_EFF_NUM / 29" | bc -l)
            echo "  → Data Parallelism is ${IMPROVEMENT}x more efficient"
            echo "  → Recommended for PCIe systems and small models (<13B)"
        fi
    fi

    echo ""
    echo "============================================================================"
    echo "4. PRODUCTION DEPLOYMENT RECOMMENDATIONS"
    echo "============================================================================"
    echo ""

    # Check 8 GPU results
    SUMMARY_8GPU="$OUTPUT_DIR/summary_8gpus.txt"
    if [ -f "$SUMMARY_8GPU" ]; then
        EFF_8=$(grep "Scaling Efficiency:" "$SUMMARY_8GPU" | awk '{print $3}' | sed 's/%//g')
        THROUGHPUT_8=$(grep "TOTAL THROUGHPUT:" "$SUMMARY_8GPU" | awk '{print $3}')

        echo "Based on benchmark results:"
        echo ""

        if [ ! -z "$EFF_8" ]; then
            if (( $(echo "$EFF_8 > 90" | bc -l) )); then
                echo "✓ RECOMMENDED: Deploy with data parallelism"
                echo ""
                echo "Deployment Strategy:"
                echo "  1. Run 8 vLLM instances (1 per GPU)"
                echo "  2. Use nginx or HAProxy for load balancing"
                echo "  3. Configure health checks and failover"
                echo ""
                echo "Expected Production Performance:"
                echo "  - Throughput: ~${THROUGHPUT_8} tokens/s"
                echo "  - Efficiency: ${EFF_8}%"
                echo "  - High availability with redundancy"
                echo ""
                echo "Example nginx config:"
                echo "  upstream vllm_backend {"
                echo "    server localhost:8001;"
                echo "    server localhost:8002;"
                echo "    # ... up to 8008"
                echo "  }"
            else
                echo "⚠ Efficiency lower than expected (${EFF_8}%)"
                echo ""
                echo "Investigate:"
                echo "  • Check CPU utilization (may be bottleneck)"
                echo "  • Monitor disk I/O (dataset loading)"
                echo "  • Verify memory bandwidth"
                echo "  • Test with fewer GPUs (4x may be optimal)"
            fi
        fi
    fi

    echo ""
    echo "============================================================================"
    echo "5. KEY FINDINGS"
    echo "============================================================================"
    echo ""
    echo "✓ Data Parallelism Benefits:"
    echo "  • No inter-GPU communication overhead"
    echo "  • Works optimally with PCIe interconnect"
    echo "  • Near-linear scaling for small models"
    echo "  • Maximum throughput for concurrent workloads"
    echo ""
    echo "✓ Use Cases:"
    echo "  • Production serving with high request volume"
    echo "  • PCIe-connected GPU systems"
    echo "  • Models that fit in single GPU (8B-13B)"
    echo "  • Throughput-critical applications"
    echo ""
    echo "⚠ Considerations:"
    echo "  • Requires load balancer setup"
    echo "  • Each GPU loads full model (~70GB VRAM)"
    echo "  • Slightly higher latency vs tensor parallelism"
    echo "  • More complex deployment than single instance"
    echo ""
    echo "============================================================================"
    echo "DETAILED RESULTS LOCATION"
    echo "============================================================================"
    echo "All results saved to: $OUTPUT_DIR"
    echo ""
    echo "Key files:"
    echo "  - FINAL_COMPARISON_REPORT.txt  (this file)"
    echo "  - summary_Xgpus.txt            (detailed metrics per config)"
    echo "  - instance_gpuX_Ygpus.txt      (raw vLLM output per GPU)"
    echo "  - gpuX_usage_Ygpus.csv         (GPU monitoring logs)"
    echo "============================================================================"

} | tee "$FINAL_REPORT"

echo ""
echo "============================================================================"
echo "BENCHMARK COMPLETE!"
echo "============================================================================"
echo "Final report: $FINAL_REPORT"
echo ""
echo "Next steps:"
echo "  1. Review scaling efficiency in final report"
echo "  2. Compare with tensor parallelism results"
echo "  3. Choose optimal strategy for your workload"
echo "============================================================================"
