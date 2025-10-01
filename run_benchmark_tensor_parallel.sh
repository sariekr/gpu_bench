#!/bin/bash

# ============================================================================
# TENSOR PARALLELISM BENCHMARK
# ============================================================================
# Objectives:
# 1. Compare throughput & latency across 1x, 2x, 4x, 8x GPU configs
# 2. Measure tensor parallelism scaling efficiency
# 3. Monitor inter-GPU communication (NVLink/PCIe)
# 4. Assess memory distribution balance
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
GPU_MEMORY_UTILIZATION=0.85
DATASET_PATH="/workspace/data/ShareGPT_V3_unfiltered_cleaned_split.json"
OUTPUT_DIR="/workspace/results/tensor_parallel_$(date +%F_%H-%M-%S)_${MODEL_NAME//\//_}"

# --- TENSOR PARALLELISM CONFIGURATION ---
# Batch sizes optimized for TP
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
export NCCL_DEBUG=INFO  # For communication debugging
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

# --- CREATE OUTPUT DIRECTORY ---
mkdir -p "$OUTPUT_DIR"
chmod +x ./monitor_gpu.sh

echo "============================================================================"
echo "TENSOR PARALLELISM BENCHMARK"
echo "============================================================================"
echo "Model: $MODEL_NAME"
echo "GPU Configurations: ${GPU_CONFIGS[@]}"
echo "Prompts: $NUM_PROMPTS"
echo "Output: $OUTPUT_DIR"
echo "============================================================================"
echo ""

# --- GPU TOPOLOGY ANALYSIS ---
if [ "$GPU_BRAND" = "nvidia" ]; then
    echo "--- GPU Topology Analysis ---"
    nvidia-smi topo -m > "$OUTPUT_DIR/gpu_topology.txt"

    # Parse topology for NVLink/PCIe info
    {
        echo "========== GPU TOPOLOGY =========="
        echo "Date: $(date)"
        echo ""
        cat "$OUTPUT_DIR/gpu_topology.txt"
        echo ""
        echo "Legend:"
        echo "  NV# = NVLink (900 GB/s per link on H100)"
        echo "  SYS = PCIe connection"
        echo "  NODE = Same NUMA node"
        echo "=================================="
    } | tee "$OUTPUT_DIR/topology_summary.txt"

    echo ""
fi

# --- BENCHMARK LOOP ---
declare -A BASELINE_THROUGHPUT
declare -A BASELINE_LATENCY

for GPUS in "${GPU_CONFIGS[@]}"
do
    echo ""
    echo "=========================================================================="
    echo "BENCHMARKING: ${GPUS}x GPU TENSOR PARALLELISM"
    echo "=========================================================================="

    OPTIMAL_MAX_SEQ=${MAX_NUM_SEQS[$GPUS]}
    echo "Configuration:"
    echo "  - Tensor Parallel Size: $GPUS"
    echo "  - Max Num Seqs (batch): $OPTIMAL_MAX_SEQ"
    echo "  - Total Prompts: $NUM_PROMPTS"
    echo ""

    # --- MONITORING SETUP ---
    MONITOR_LOG_FILE="$OUTPUT_DIR/gpu_metrics_${GPUS}gpus.csv"
    NCCL_LOG_FILE="$OUTPUT_DIR/nccl_${GPUS}gpus.log"

    # Enhanced GPU monitoring script
    cat > "$OUTPUT_DIR/monitor_enhanced_${GPUS}gpus.sh" << 'EOF_MONITOR'
#!/bin/bash
OUTPUT_FILE=$1
echo "timestamp,gpu_id,utilization_%,memory_used_GB,memory_total_GB,temperature_C,power_W" > "$OUTPUT_FILE"

while true; do
    nvidia-smi --query-gpu=timestamp,gpu_uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits | \
    awk -F, '{printf "%s,%s,%.1f,%.2f,%.2f,%s,%s\n", $1, $2, $3, $4/1024, $5/1024, $6, $7}' >> "$OUTPUT_FILE"
    sleep 0.5
done
EOF_MONITOR

    chmod +x "$OUTPUT_DIR/monitor_enhanced_${GPUS}gpus.sh"

    # Start monitoring
    "$OUTPUT_DIR/monitor_enhanced_${GPUS}gpus.sh" "$MONITOR_LOG_FILE" &
    MONITOR_PID=$!
    sleep 2

    if [ "$GPU_BRAND" = "nvidia" ]; then
        # --- WARMUP ---
        echo "Phase 1: Warmup..."
        vllm bench throughput \
            --model "$MODEL_NAME" \
            --dataset-name sharegpt \
            --dataset-path "$DATASET_PATH" \
            --tensor-parallel-size "$GPUS" \
            --num-prompts 500 \
            --max-num-seqs "$OPTIMAL_MAX_SEQ" \
            --max-model-len 4096 \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" > /dev/null 2>&1

        echo "Warmup completed."
        echo ""

        # --- ACTUAL BENCHMARK ---
        echo "Phase 2: Running benchmark..."
        RAW_RESULT_FILE="$OUTPUT_DIR/throughput_raw_${GPUS}gpus.txt"

        START_TIME=$(date +%s.%N)

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
            --seed 42 2>&1 | tee "$RAW_RESULT_FILE"

        END_TIME=$(date +%s.%N)
        TOTAL_TIME=$(echo "$END_TIME - $START_TIME" | bc -l)

        # Stop monitoring
        kill $MONITOR_PID 2>/dev/null

        echo ""
        echo "--- Phase 3: Analyzing Results ---"

        # --- PARSE RESULTS ---
        if [ ! -f "$RAW_RESULT_FILE" ]; then
            echo "ERROR: Raw result file not found!"
            continue
        fi

        RESULT_LINE=$(grep "Throughput:" "$RAW_RESULT_FILE" 2>/dev/null)

        if [ -z "$RESULT_LINE" ]; then
            echo "ERROR: Throughput data not found!"
            echo "Last 30 lines of output:"
            tail -30 "$RAW_RESULT_FILE"
            continue
        fi

        # Extract metrics
        REQUESTS_PER_SEC=$(echo "$RESULT_LINE" | awk '{print $2}')
        TOTAL_TOKENS_PER_SEC=$(echo "$RESULT_LINE" | awk '{print $4}')

        # Calculate latency (average time per request)
        AVG_LATENCY_MS=$(echo "scale=2; 1000 / $REQUESTS_PER_SEC" | bc -l)

        # Per-GPU throughput
        THROUGHPUT_PER_GPU=$(echo "scale=2; $TOTAL_TOKENS_PER_SEC / $GPUS" | bc -l)

        # --- MEMORY ANALYSIS ---
        echo "Analyzing memory distribution..."

        # Parse monitoring CSV for peak memory
        if [ -f "$MONITOR_LOG_FILE" ]; then
            # Get peak memory for each GPU
            declare -a GPU_PEAK_MEMORY
            declare -a GPU_IDS

            # Extract unique GPU IDs
            readarray -t GPU_IDS < <(tail -n +2 "$MONITOR_LOG_FILE" | cut -d',' -f2 | sort -u)

            TOTAL_PEAK_MEMORY=0
            for i in "${!GPU_IDS[@]}"; do
                GPU_ID="${GPU_IDS[$i]}"
                PEAK=$(grep "$GPU_ID" "$MONITOR_LOG_FILE" | cut -d',' -f4 | sort -n | tail -1)
                GPU_PEAK_MEMORY[$i]=$PEAK
                TOTAL_PEAK_MEMORY=$(echo "$TOTAL_PEAK_MEMORY + $PEAK" | bc -l)
            done

            # Calculate memory balance (std deviation)
            if [ ${#GPU_PEAK_MEMORY[@]} -gt 1 ]; then
                AVG_MEMORY=$(echo "scale=2; $TOTAL_PEAK_MEMORY / ${#GPU_PEAK_MEMORY[@]}" | bc -l)

                # Calculate variance
                VARIANCE=0
                for mem in "${GPU_PEAK_MEMORY[@]}"; do
                    DIFF=$(echo "$mem - $AVG_MEMORY" | bc -l)
                    SQ=$(echo "$DIFF * $DIFF" | bc -l)
                    VARIANCE=$(echo "$VARIANCE + $SQ" | bc -l)
                done
                VARIANCE=$(echo "scale=4; $VARIANCE / ${#GPU_PEAK_MEMORY[@]}" | bc -l)
                STD_DEV=$(echo "scale=2; sqrt($VARIANCE)" | bc -l)

                # Memory balance % (100% = perfect balance)
                MEMORY_BALANCE=$(echo "scale=2; 100 - ($STD_DEV / $AVG_MEMORY * 100)" | bc -l)
            else
                AVG_MEMORY=$TOTAL_PEAK_MEMORY
                STD_DEV=0
                MEMORY_BALANCE=100
            fi
        else
            TOTAL_PEAK_MEMORY="N/A"
            MEMORY_BALANCE="N/A"
        fi

        # --- SCALING EFFICIENCY ---
        if [ "$GPUS" -eq 1 ]; then
            BASELINE_THROUGHPUT[1]=$TOTAL_TOKENS_PER_SEC
            BASELINE_LATENCY[1]=$AVG_LATENCY_MS
            SCALING_EFFICIENCY="N/A (baseline)"
            SPEEDUP="1.00x"
        else
            # Expected throughput (ideal linear scaling)
            IDEAL_THROUGHPUT=$(echo "${BASELINE_THROUGHPUT[1]} * $GPUS" | bc -l)

            # Actual speedup
            ACTUAL_SPEEDUP=$(echo "scale=2; $TOTAL_TOKENS_PER_SEC / ${BASELINE_THROUGHPUT[1]}" | bc -l)

            # Efficiency % = (actual / ideal) * 100
            SCALING_EFFICIENCY=$(echo "scale=2; ($ACTUAL_SPEEDUP / $GPUS) * 100" | bc -l)

            SPEEDUP="${ACTUAL_SPEEDUP}x"
        fi

        # --- INTER-GPU COMMUNICATION ANALYSIS ---
        if [ "$GPUS" -gt 1 ]; then
            echo "Analyzing inter-GPU communication..."

            # Extract NCCL communication stats from vLLM output
            NCCL_INIT_TIME=$(grep -i "nccl.*init" "$RAW_RESULT_FILE" | grep -oP '\d+\.\d+' | head -1 || echo "N/A")

            # Estimate communication overhead
            # Communication overhead = (Ideal time - Actual time) / Actual time
            IDEAL_TIME=$(echo "scale=4; $TOTAL_TIME / $GPUS" | bc -l)
            if (( $(echo "$TOTAL_TIME > 0" | bc -l) )); then
                COMM_OVERHEAD_PCT=$(echo "scale=2; (($TOTAL_TIME - $IDEAL_TIME) / $TOTAL_TIME) * 100" | bc -l)
            else
                COMM_OVERHEAD_PCT="N/A"
            fi

            # Check topology for connection type
            if [ -f "$OUTPUT_DIR/gpu_topology.txt" ]; then
                # Check if NVLink (NV*) or PCIe (SYS) dominant
                NVLINK_COUNT=$(grep -c "NV[0-9]" "$OUTPUT_DIR/gpu_topology.txt" || echo 0)
                PCIE_COUNT=$(grep -c "SYS" "$OUTPUT_DIR/gpu_topology.txt" || echo 0)

                if [ "$NVLINK_COUNT" -gt "$PCIE_COUNT" ]; then
                    INTERCONNECT="NVLink-dominant"
                    # H100 NVLink: ~900 GB/s per link
                    THEORETICAL_BW="900 GB/s per link"
                else
                    INTERCONNECT="PCIe-dominant"
                    # PCIe Gen4 x16: ~32 GB/s
                    THEORETICAL_BW="~32 GB/s (PCIe Gen4)"
                fi
            else
                INTERCONNECT="Unknown"
                THEORETICAL_BW="N/A"
            fi
        else
            COMM_OVERHEAD_PCT="N/A (single GPU)"
            INTERCONNECT="N/A"
            THEORETICAL_BW="N/A"
        fi

        # --- GENERATE SUMMARY REPORT ---
        SUMMARY_FILE="$OUTPUT_DIR/summary_${GPUS}gpus.txt"

        {
            echo "============================================================================"
            echo "TENSOR PARALLELISM BENCHMARK SUMMARY"
            echo "============================================================================"
            echo "Date: $(date)"
            echo "Model: $MODEL_NAME"
            echo "GPU Count: $GPUS"
            echo ""
            echo "--- Configuration ---"
            echo "Tensor Parallel Size: $GPUS"
            echo "Max Num Seqs (batch): $OPTIMAL_MAX_SEQ"
            echo "Total Prompts: $NUM_PROMPTS"
            echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
            echo "Total Execution Time: $(printf "%.2f" $TOTAL_TIME)s"
            echo ""
            echo "--- 1. THROUGHPUT & LATENCY METRICS ---"
            echo "Total Throughput: $TOTAL_TOKENS_PER_SEC tokens/s"
            echo "Throughput per GPU: $THROUGHPUT_PER_GPU tokens/s"
            echo "Requests per Second: $REQUESTS_PER_SEC req/s"
            echo "Average Latency: $AVG_LATENCY_MS ms"
            echo ""
            echo "--- 2. SCALING EFFICIENCY ---"
            echo "Speedup vs 1 GPU: $SPEEDUP"
            echo "Scaling Efficiency: $SCALING_EFFICIENCY%"
            if [ "$GPUS" -gt 1 ]; then
                echo "  (100% = perfect linear scaling)"
                echo "  (Actual: $(printf "%.1f" $ACTUAL_SPEEDUP)x out of ideal ${GPUS}.0x)"
            fi
            echo ""
            echo "--- 3. INTER-GPU COMMUNICATION ---"
            if [ "$GPUS" -gt 1 ]; then
                echo "Interconnect Type: $INTERCONNECT"
                echo "Theoretical Bandwidth: $THEORETICAL_BW"
                echo "Communication Overhead: $COMM_OVERHEAD_PCT%"
                echo "  (Time spent in inter-GPU sync vs computation)"
            else
                echo "N/A (single GPU configuration)"
            fi
            echo ""
            echo "--- 4. MEMORY DISTRIBUTION ---"
            echo "Total Peak Memory: $(printf "%.2f" $TOTAL_PEAK_MEMORY) GB"
            if [ ${#GPU_PEAK_MEMORY[@]} -gt 0 ]; then
                echo "Peak Memory per GPU:"
                for i in "${!GPU_PEAK_MEMORY[@]}"; do
                    echo "  GPU $i: $(printf "%.2f" ${GPU_PEAK_MEMORY[$i]}) GB"
                done
                echo "Average Memory per GPU: $(printf "%.2f" $AVG_MEMORY) GB"
                echo "Memory Std Deviation: $(printf "%.2f" $STD_DEV) GB"
                echo "Memory Balance: $MEMORY_BALANCE%"
                echo "  (100% = perfectly balanced, >95% = excellent)"
            fi
            echo ""
            echo "============================================================================"
        } | tee "$SUMMARY_FILE"

        echo ""
        echo "Summary saved to: $SUMMARY_FILE"
        echo ""

    fi
done

# --- FINAL COMPARISON REPORT ---
echo ""
echo "============================================================================"
echo "GENERATING FINAL COMPARISON REPORT"
echo "============================================================================"

FINAL_REPORT="$OUTPUT_DIR/FINAL_COMPARISON_REPORT.txt"

{
    echo "============================================================================"
    echo "MULTI-GPU TENSOR PARALLELISM SCALING ANALYSIS"
    echo "============================================================================"
    echo "Model: $MODEL_NAME"
    echo "Date: $(date)"
    echo "GPU Brand: $GPU_BRAND"
    echo ""
    echo "============================================================================"
    echo "1. THROUGHPUT & LATENCY COMPARISON"
    echo "============================================================================"
    printf "%-10s | %-18s | %-18s | %-15s\n" \
        "GPU Count" "Total Throughput" "Throughput/GPU" "Avg Latency"
    echo "-----------|--------------------|--------------------|----------------"

    for GPUS in "${GPU_CONFIGS[@]}"; do
        SUMMARY_FILE="$OUTPUT_DIR/summary_${GPUS}gpus.txt"
        if [ -f "$SUMMARY_FILE" ]; then
            THROUGHPUT=$(grep "Total Throughput:" "$SUMMARY_FILE" | awk '{print $3, $4}')
            PER_GPU=$(grep "Throughput per GPU:" "$SUMMARY_FILE" | awk '{print $4, $5}')
            LATENCY=$(grep "Average Latency:" "$SUMMARY_FILE" | awk '{print $3, $4}')

            printf "%-10s | %-18s | %-18s | %-15s\n" \
                "${GPUS}x" "$THROUGHPUT" "$PER_GPU" "$LATENCY"
        fi
    done

    echo ""
    echo "============================================================================"
    echo "2. SCALING EFFICIENCY ANALYSIS"
    echo "============================================================================"
    printf "%-10s | %-12s | %-20s | %-15s\n" \
        "GPU Count" "Speedup" "Scaling Efficiency" "Status"
    echo "-----------|--------------|----------------------|----------------"

    for GPUS in "${GPU_CONFIGS[@]}"; do
        SUMMARY_FILE="$OUTPUT_DIR/summary_${GPUS}gpus.txt"
        if [ -f "$SUMMARY_FILE" ]; then
            SPEEDUP=$(grep "Speedup vs 1 GPU:" "$SUMMARY_FILE" | awk '{print $5}')
            EFFICIENCY=$(grep "Scaling Efficiency:" "$SUMMARY_FILE" | awk '{print $3}')

            # Determine status
            if [ "$GPUS" -eq 1 ]; then
                STATUS="Baseline"
            else
                EFF_NUM=$(echo "$EFFICIENCY" | sed 's/%//g')
                if (( $(echo "$EFF_NUM >= 90" | bc -l) )); then
                    STATUS="Excellent"
                elif (( $(echo "$EFF_NUM >= 70" | bc -l) )); then
                    STATUS="Good"
                elif (( $(echo "$EFF_NUM >= 50" | bc -l) )); then
                    STATUS="Fair"
                else
                    STATUS="Poor"
                fi
            fi

            printf "%-10s | %-12s | %-20s | %-15s\n" \
                "${GPUS}x" "$SPEEDUP" "$EFFICIENCY" "$STATUS"
        fi
    done

    echo ""
    echo "Interpretation:"
    echo "  Excellent (≥90%): Near-perfect scaling, minimal overhead"
    echo "  Good (70-89%):    Acceptable scaling, some communication overhead"
    echo "  Fair (50-69%):    Moderate overhead, consider optimization"
    echo "  Poor (<50%):      Significant overhead, may not justify GPU count"
    echo ""

    echo "============================================================================"
    echo "3. INTER-GPU COMMUNICATION SUMMARY"
    echo "============================================================================"

    if [ -f "$OUTPUT_DIR/topology_summary.txt" ]; then
        echo "GPU Topology:"
        grep -A 20 "Legend:" "$OUTPUT_DIR/topology_summary.txt" | head -15
        echo ""
    fi

    printf "%-10s | %-20s | %-25s\n" \
        "GPU Count" "Interconnect" "Comm Overhead %"
    echo "-----------|----------------------|---------------------------"

    for GPUS in "${GPU_CONFIGS[@]}"; do
        if [ "$GPUS" -gt 1 ]; then
            SUMMARY_FILE="$OUTPUT_DIR/summary_${GPUS}gpus.txt"
            if [ -f "$SUMMARY_FILE" ]; then
                INTERCONNECT=$(grep "Interconnect Type:" "$SUMMARY_FILE" | cut -d':' -f2 | xargs)
                OVERHEAD=$(grep "Communication Overhead:" "$SUMMARY_FILE" | awk '{print $3}')

                printf "%-10s | %-20s | %-25s\n" \
                    "${GPUS}x" "$INTERCONNECT" "$OVERHEAD"
            fi
        fi
    done

    echo ""

    echo "============================================================================"
    echo "4. MEMORY DISTRIBUTION SUMMARY"
    echo "============================================================================"
    printf "%-10s | %-18s | %-18s | %-15s\n" \
        "GPU Count" "Total Peak Mem" "Avg per GPU" "Balance %"
    echo "-----------|--------------------|--------------------|----------------"

    for GPUS in "${GPU_CONFIGS[@]}"; do
        SUMMARY_FILE="$OUTPUT_DIR/summary_${GPUS}gpus.txt"
        if [ -f "$SUMMARY_FILE" ]; then
            TOTAL_MEM=$(grep "Total Peak Memory:" "$SUMMARY_FILE" | awk '{print $4, $5}')
            AVG_MEM=$(grep "Average Memory per GPU:" "$SUMMARY_FILE" | awk '{print $5, $6}')
            BALANCE=$(grep "Memory Balance:" "$SUMMARY_FILE" | awk '{print $3}')

            printf "%-10s | %-18s | %-18s | %-15s\n" \
                "${GPUS}x" "$TOTAL_MEM" "$AVG_MEM" "$BALANCE"
        fi
    done

    echo ""
    echo "Interpretation:"
    echo "  >95%:  Excellent balance - optimal tensor sharding"
    echo "  85-95%: Good balance - minor imbalances"
    echo "  <85%:  Poor balance - check tensor distribution"
    echo ""

    echo "============================================================================"
    echo "KEY FINDINGS & RECOMMENDATIONS"
    echo "============================================================================"
    echo ""

    # Auto-generate recommendations based on results
    echo "Based on the benchmark results:"
    echo ""

    # Check 4x GPU efficiency
    if [ -f "$OUTPUT_DIR/summary_4gpus.txt" ]; then
        EFF_4=$(grep "Scaling Efficiency:" "$OUTPUT_DIR/summary_4gpus.txt" | awk '{print $3}' | sed 's/%//g')

        if (( $(echo "$EFF_4 >= 70" | bc -l) )); then
            echo "✓ 4x GPU configuration shows good scaling (${EFF_4}%)"
            echo "  → Tensor parallelism is effective for this model size"
        else
            echo "✗ 4x GPU configuration shows poor scaling (${EFF_4}%)"
            echo "  → Consider data parallelism instead for better throughput"
            echo "  → Model may be too small for effective tensor sharding"
        fi
    fi

    echo ""
    echo "For optimal multi-GPU deployment:"
    echo "  1. Review scaling efficiency - aim for >70% on target GPU count"
    echo "  2. Check memory balance - should be >95% for even distribution"
    echo "  3. Monitor communication overhead - high overhead indicates bottleneck"
    echo "  4. Compare with data parallelism approach for small models (<13B)"
    echo ""

    echo "============================================================================"
    echo "DETAILED RESULTS LOCATION"
    echo "============================================================================"
    echo "All results saved to: $OUTPUT_DIR"
    echo ""
    echo "Key files:"
    echo "  - FINAL_COMPARISON_REPORT.txt  (this file)"
    echo "  - summary_Xgpus.txt            (detailed metrics per config)"
    echo "  - gpu_topology.txt             (hardware interconnect info)"
    echo "  - gpu_metrics_Xgpus.csv        (time-series GPU utilization)"
    echo "============================================================================"

} | tee "$FINAL_REPORT"

echo ""
echo "============================================================================"
echo "BENCHMARK COMPLETE!"
echo "============================================================================"
echo "Final report: $FINAL_REPORT"
echo "============================================================================"
