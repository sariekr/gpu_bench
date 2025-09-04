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
    # --- EN BASİT VE EN GARANTİLİ AMD KOMUTU ---
    # Sadece VRAM kullanımını al, 'M' harfini sil, GB'a çevir.
    # Hata kontrolü de ekledik.
    TIMESTAMP=$(date '+%Y/%m/%d %H:%M:%S.%3N')
    # '--showmemuse' sadece bellek kullanımını verir.
    # grep 'GPU[0]' ile sadece ilk GPU'nun satırını alıyoruz.
    # awk '{print $5}' ile sadece 5. sütunu (örn: 512M) alıyoruz.
    MEM_USED_RAW=$(rocm-smi --showmemuse | grep 'GPU[0]' | awk '{print $5}')
    
    # Gelen verinin sayı içerip içermediğini kontrol et
    if [[ $MEM_USED_RAW =~ [0-9] ]]; then
        MEM_USED_MB=$(echo "$MEM_USED_RAW" | sed 's/M//g')
        MEM_USED_GB=$(echo "scale=3; $MEM_USED_MB / 1024" | bc)
        echo "$TIMESTAMP,GPU0,$MEM_USED_GB" >> "$OUTPUT_FILE"
    fi
    sleep 0.1
  done
fi