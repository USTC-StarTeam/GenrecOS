#!/bin/bash

# 1. 定义时间戳变量 (格式: YYYYMMDD_HHMMSS)
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

# 2. 确保日志目录存在 (防止报错)
LOG_DIR="temp_try_GRPO_Rec_Output/logs"
mkdir -p "$LOG_DIR"

# 3. 运行命令，日志文件名包含时间戳
# 结果类似: eval_test_20260115_221030.log
python -u try_train_grpo.py --config ./rl_configs/Beauty_llamarec_DIN.yaml 2>&1 | tee -a "$LOG_DIR/rl_grpo_$TIMESTAMP.log"