#!/bin/bash

# ===================== 核心配置 =====================

TARGET_GPUS="0,1,2,3"

PROJECT_ROOT="$(pwd)"
cd "${PROJECT_ROOT}"

MODEL_PATH="../models/cot_model/best_model"
DATA_DIR="../../Data/Beauty/FT_data/beauty_processed"
TEST_DATA_ORIGIN="${DATA_DIR}/cot_data/test.json" 
TRIE_PATH="${DATA_DIR}/global_trie.pkl"


JOB_TAG="$(date +%Y%m%d_%H%M%S)_$$"
TEMP_SHARD_DIR="./temp_shards_${JOB_TAG}"
mkdir -p ${TEMP_SHARD_DIR}

echo "Job Tag: $JOB_TAG"
echo "Starting PARALLEL evaluation..."
echo "Target GPUs: ${TARGET_GPUS}"

# ===================== 自动并行逻辑 =====================

IFS=',' read -r -a GPU_ARRAY <<< "$TARGET_GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "Detected ${NUM_GPUS} GPUs configured for this job: ${GPU_ARRAY[*]}"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Error: No GPUs configured in TARGET_GPUS variable!"
    exit 1
fi


echo "Splitting data into ${NUM_GPUS} parts..."
TOTAL_LINES=$(wc -l < "${TEST_DATA_ORIGIN}")
LINES_PER_GPU=$(( ($TOTAL_LINES + $NUM_GPUS - 1) / $NUM_GPUS ))


split -l ${LINES_PER_GPU} -d --additional-suffix=.json "${TEST_DATA_ORIGIN}" "${TEMP_SHARD_DIR}/shard_"


pids=()

for ((i=0; i<NUM_GPUS; i++)); do
    # 获取真实的物理 GPU ID
    REAL_GPU_ID=${GPU_ARRAY[$i]}
    
    # 构造分片文件名 (注意 split 生成的后缀总是从 00 开始，所以这里用 $i 是对的)
    SHARD_FILE="${TEMP_SHARD_DIR}/shard_$(printf "%02d" $i).json"
    # 日志名带上物理 GPU ID 方便排查
    GPU_LOG="${TEMP_SHARD_DIR}/eval_gpu_${REAL_GPU_ID}.log"
    
    if [ ! -f "$SHARD_FILE" ]; then
        echo "Warning: Shard file $SHARD_FILE not found, skipping GPU $REAL_GPU_ID"
        continue
    fi

    echo "Running on Physical GPU ${REAL_GPU_ID} (Shard $i) > ${GPU_LOG}"

    
    CUDA_VISIBLE_DEVICES=$REAL_GPU_ID python result_generate.py \
        --model_path "${MODEL_PATH}" \
        --data_path "${SHARD_FILE}" \
        --trie_path "${TRIE_PATH}" \
        --device "cuda" \
        --think_max_tokens 128 \
        --think_temperature 1.5 \
        --think_top_p 0.95 \
        --num_think_samples 5 \
        --sid_max_tokens 10 \
        --sid_temperature 0.6 \
        --sid_top_p 1.0 \
        --num_sid_beams 10 \
        > "${GPU_LOG}" 2>&1 &
    
    pids+=($!)
done

# 4. 等待
echo "All tasks launched. Waiting for completion..."
wait

echo "All GPU tasks finished."

# ===================== 结果简报 =====================
echo "================ Summary ================"
for ((i=0; i<NUM_GPUS; i++)); do
    REAL_GPU_ID=${GPU_ARRAY[$i]}
    GPU_LOG="${TEMP_SHARD_DIR}/eval_gpu_${REAL_GPU_ID}.log"
    echo "--- GPU $REAL_GPU_ID Results ---"
    if [ -f "$GPU_LOG" ]; then
        grep -A 2 "Final Average" "${GPU_LOG}" || echo "Metrics not found in ${GPU_LOG}"
    else
        echo "Log file not found."
    fi
done
echo "========================================="

# rm -rf ${TEMP_SHARD_DIR}
echo "Done."