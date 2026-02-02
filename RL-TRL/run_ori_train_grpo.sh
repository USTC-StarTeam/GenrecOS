#!/bin/bash

# 1. 定义时间戳变量
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

# 2. 确保日志目录存在
LOG_DIR="temp_try_GRPO_Rec_Output/logs"
mkdir -p "$LOG_DIR"

# 定义日志文件路径（提取出来方便后面引用）
LOG_FILE="$LOG_DIR/rl_grpo_$TIMESTAMP.log"

# ================= 改动部分 =================

# 3. 将具体的 Python 命令定义为变量
CMD="python -u ori_train_grpo.py --config ./rl_configs/KuaiRec_big_qwen2_DIN.yaml"

# 4. 先将命令写入日志文件开头
echo "Execution Command: $CMD" > "$LOG_FILE"
echo "Start Time: $TIMESTAMP" >> "$LOG_FILE"
echo "------------------------------------------------------" >> "$LOG_FILE"

# 5. 执行命令，并将输出追加到日志 (注意这里使用 $CMD)
# 2>&1 将错误输出也合并，tee -a 追加到文件
$CMD 2>&1 | tee -a "$LOG_FILE"