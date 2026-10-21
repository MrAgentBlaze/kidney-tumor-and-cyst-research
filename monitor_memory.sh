#!/usr/bin/env bash

set -u

OUT="${1:-memory_monitor_$(date +%Y%m%d_%H%M%S).csv}"

echo "timestamp,ram_total_mib,ram_used_mib,ram_available_mib,swap_used_mib,gpu_used_mib,gpu_total_mib,gpu_util_percent" > "$OUT"

echo "Logging memory to: $OUT"
echo "Press Ctrl+C to stop."

while true; do
    timestamp="$(date --iso-8601=seconds)"

    read -r ram_total ram_used ram_available swap_used < <(
        free -m | awk '
        /^Mem:/  { total=$2; used=$3; available=$7 }
        /^Swap:/ { swap=$3 }
        END { print total, used, available, swap }
        '
    )

    gpu_values="$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"

    if [[ -n "$gpu_values" ]]; then
        IFS=',' read -r gpu_used gpu_total gpu_util <<< "$gpu_values"
        gpu_used="${gpu_used// /}"
        gpu_total="${gpu_total// /}"
        gpu_util="${gpu_util// /}"
    else
        gpu_used=""
        gpu_total=""
        gpu_util=""
    fi

    echo "$timestamp,$ram_total,$ram_used,$ram_available,$swap_used,$gpu_used,$gpu_total,$gpu_util" >> "$OUT"
    sleep 2
done
