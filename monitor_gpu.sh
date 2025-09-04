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
    # --showmemused yerine daha uyumlu olan --showmemuse kullanıldı.
    rocm-smi --showmemuse -d 0 | tail -n 1 | awk '{print strftime("%Y/%m/%d %H:%M:%S.000") ",0," $5/1024}' >> "$OUTPUT_FILE"
    sleep 0.1
  done
fi