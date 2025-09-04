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
MODEL_NAME="openai/gpt-oss-20b"
GPU_CONFIGS=(1)
NUM_PROMPTS=1000
# GPU Memory Utilization, AMD dökümanında 0.9 olarak önerilmiş.
GPU_MEMORY_UTILIZATION=0.9
DATASET_PATH="/workspace/data/ShareGPT_V3_unfiltered_cleaned_split.json"
OUTPUT_DIR="/workspace/results/benchmark_$(date +%F_%H-%M-%S)_${MODEL_NAME//\//_}"



# --- BENCHMARK BAŞLANGICI ---
mkdir -p "$OUTPUT_DIR"
chmod +x ./monitor_gpu.sh

echo "### Starting Benchmark Run ###"
echo "Model: $MODEL_NAME"
echo "GPU Configs: ${GPU_CONFIGS[@]}"
echo "Num Prompts: $NUM_PROMPTS"
echo "GPU Memory Utilization (AMD): $GPU_MEMORY_UTILIZATION"
echo "Results will be saved in: $OUTPUT_DIR"

for GPUS in "${GPU_CONFIGS[@]}"
do
  echo ""
  echo "--- Running Benchmark for $GPUS GPU(s) on $GPU_BRAND platform ---"
  
  # Ortak Hazırlık
  MONITOR_LOG_FILE="$OUTPUT_DIR/gpu_usage_${GPUS}_gpus.csv"
  RAW_RESULT_FILE="$OUTPUT_DIR/throughput_raw_${GPUS}_gpus.txt"
  ./monitor_gpu.sh "$MONITOR_LOG_FILE" "$GPU_BRAND" &
  MONITOR_PID=$!
  sleep 2

  # Platforma Özel Komutları Çalıştır
  if [ "$GPU_BRAND" = "nvidia" ]; then
    echo "INFO: Running NVIDIA optimized command..."
    vllm bench throughput \
      --model "$MODEL_NAME" \
      --dataset-name sharegpt \
      --dataset-path "$DATASET_PATH" \
      --tensor-parallel-size "$GPUS" \
      --num-prompts "$NUM_PROMPTS" \
      --gpu-memory-utilization 0.9 > "$RAW_RESULT_FILE"
  
  elif [ "$GPU_BRAND" = "amd" ]; then
    echo "INFO: Running AMD official optimized command..."
    # --- AMD İÇİN KRİTİK PERFORMANS AYARLARI ---
    export VLLM_USE_TRITON_FLASH_ATTN=0
    export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1
    
    # Dökümandaki optimize edilmiş komut
    python3 /app/vllm/benchmarks/benchmark_throughput.py \
        --model "$MODEL_NAME" \
        --dataset-name sharegpt \
        --dataset-path "$DATASET_PATH" \
        --tensor-parallel-size "$GPUS" \
        --num-prompts "$NUM_PROMPTS" \
        --max-num-seqs "$NUM_PROMPTS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --dtype float16 \
        --max-model-len 8192 > "$RAW_RESULT_FILE"
  fi

  kill $MONITOR_PID
  echo "--- Raw Benchmark completed. Starting post-processing... ---"

  # --- SONUÇLARI İŞLEME VE HESAPLAMA ---
  RESULT_LINE=$(grep "Throughput:" "$RAW_RESULT_FILE")
  if [ -z "$RESULT_LINE" ]; then
    echo "ERROR: Benchmark failed. Could not find 'Throughput:' line in raw results."
    cat "$RAW_RESULT_FILE"
    continue # Döngünün bir sonraki adımına geç
  fi

  REQUESTS_PER_SEC=$(echo "$RESULT_LINE" | awk '{print $2}')
  TOTAL_TOKENS_PER_SEC=$(echo "$RESULT_LINE" | awk '{print $4}')

  # 1. Basic Performance Metriklerini Hesapla
  THROUGHPUT_PER_GPU=$(echo "scale=2; $TOTAL_TOKENS_PER_SEC / $GPUS" | bc -l)
  AVG_LATENCY_MS=$(echo "scale=2; 1000 / $REQUESTS_PER_SEC" | bc -l)

  # 3. Memory Assessment Metriklerini Hesapla
  PEAK_MEM_PER_GPU_GB=$(tail -n +2 "$MONITOR_LOG_FILE" | awk -F, '{ if ($3 > max) { max = $3 } } END { print max }')
  PEAK_MEM_TOTAL_GB=$(echo "scale=2; $PEAK_MEM_PER_GPU_GB * $GPUS" | bc -l)

  MEM_BALANCE_STATS=$(tail -n +2 "$MONITOR_LOG_FILE" | awk -F, '
    BEGIN { first_timestamp = ""; count = 0; }
    {
        if (first_timestamp == "") { first_timestamp = $1; }
        if ($1 == first_timestamp) {
            values[count] = $3;
            sum += $3;
            sumsq += $3^2;
            count++;
        }
    }
    END {
        if (count > 0) {
            mean = sum / count;
            stdev = sqrt(sumsq/count - mean^2);
            if (mean > 0) {
                balance = (1 - (stdev/mean)) * 100;
            } else {
                balance = 0;
            }
            printf "%.2f", balance;
        } else {
            printf "0.00";
        }
    }')
  if [ "$GPUS" -eq 1 ]; then MEM_BALANCE_STATS="100.00"; fi

  # Okunabilir özet dosyasını oluştur
  {
    echo "========== BENCHMARK SUMMARY =========="
    echo "Date: $(date)"
    echo "Model: $MODEL_NAME"
    echo "GPU Count: $GPUS"
    echo ""
    echo "--- 1. Basic Performance Comparison ---"
    echo "Throughput (tokens/second/GPU): $THROUGHPUT_PER_GPU"
    echo "Average Inference Latency (milliseconds): $AVG_LATENCY_MS"
    echo ""
    echo "--- 2. Tensor Parallelism Efficiency Analysis ---"
    echo "NOTE: Scaling efficiency requires a 1-GPU baseline run for comparison."
    echo ""
    echo "--- 3. Memory Distribution Assessment ---"
    echo "Peak Memory Usage per GPU (gigabytes): $PEAK_MEM_PER_GPU_GB"
    echo "Total Peak Memory Usage (gigabytes): $PEAK_MEM_TOTAL_GB"
    echo "Memory Distribution Balance (%): $MEM_BALANCE_STATS"
    echo ""
    echo "--- Raw Data ---"
    echo "Total Throughput: $TOTAL_TOKENS_PER_SEC total tokens/s"
    echo "Requests per Second: $REQUESTS_PER_SEC requests/s"
    echo "======================================="
  } > "$SUMMARY_FILE"

  # JSON formatında özet oluştur
  {
  echo "{"
  echo "  \"model_name\": \"$MODEL_NAME\","
  echo "  \"gpu_count\": $GPUS,"
  echo "  \"throughput_tokens_per_second_per_gpu\": $THROUGHPUT_PER_GPU,"
  echo "  \"average_inference_latency_ms\": $AVG_LATENCY_MS,"
  echo "  \"peak_memory_usage_per_gpu_gb\": $PEAK_MEM_PER_GPU_GB,"
  echo "  \"total_peak_memory_usage_gb\": $PEAK_MEM_TOTAL_GB,"
  echo "  \"memory_distribution_balance_percent\": $MEM_BALANCE_STATS"
  echo "}"
  } > "$JSON_SUMMARY_FILE"


  echo "--- Post-processing finished. Summary written to $SUMMARY_FILE ---"
done

echo ""
echo "### Benchmark Run Finished! ###"
echo "To see the detailed summary, run:"
# İlk GPU yapılandırmasının özetini göster
FIRST_GPU_CONFIG=$(echo "${GPU_CONFIGS[0]}")
echo "cat $OUTPUT_DIR/summary_results_${FIRST_GPU_CONFIG}_gpus.txt"