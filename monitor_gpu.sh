#!/bin/bash
OUTPUT_FILE=$1
GPU_TYPE=$2
echo "timestamp,gpu_id,memory.used_GB" > "$OUTPUT_FILE"

if [ "$GPU_TYPE" = "nvidia" ]; then
  while true; do
    nvidia-smi --query-gpu=timestamp,gpu_uuid,memory.used --format=csv,noheader,nounits | awk -F, '{print $1 "," $2 "," $3/1024}' >> "$OUTPUT_FILE"
    sleep 0.1
  done
else # amd
  while true; do
    # --- EN SAĞLAM AMD KOMUTU ---
    # Sadece VRAM kullanımını alıp, 'M' harfini silip, GB'a çeviriyoruz.
    TIMESTAMP=$(date '+%Y/%m/%d %H:%M:%S.%3N')
    MEM_USED_MB=$(rocm-smi -u | grep 'GPU' | awk '{print $5}' | sed 's/M//g')
    # Eğer MEM_USED_MB boş değilse, hesapla ve yaz.
    if [ ! -z "$MEM_USED_MB" ]; then
      MEM_USED_GB=$(echo "scale=3; $MEM_USED_MB / 1024" | bc)
      echo "$TIMESTAMP,GPU0,$MEM_USED_GB" >> "$OUTPUT_FILE"
    fi
    sleep 0.1
  done
fi