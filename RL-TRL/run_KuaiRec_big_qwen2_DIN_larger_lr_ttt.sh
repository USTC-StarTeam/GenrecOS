#!/bin/bash

# Usage: sh run_train_grpo.sh <python_file_name>

# ================= 改动部分 1: 获取命令行参数 =================
# 获取第一个参数作为 Python 文件名
PYTHON_SCRIPT=$1

# 检查是否输入了文件名，如果没有则提示用法并退出
if [ -z "$PYTHON_SCRIPT" ]; then
    echo "错误: 未指定 Python 文件名。"
    echo "用法: $0 <python_file_name>"
    echo "示例: $0 temp_train_grpo_3_wodin.py"
    exit 1
fi
# ==========================================================

# 1. 定义时间戳变量
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

# 2. 确保日志目录存在
LOG_DIR="temp_try_GRPO_Rec_Output/logs"
mkdir -p "$LOG_DIR"

# 定义日志文件路径（文件名中包含脚本名，方便区分）
# 使用 basename 去除路径，只保留文件名（例如 path/to/script.py -> script.py）
SCRIPT_BASENAME=$(basename "$PYTHON_SCRIPT")
LOG_FILE="$LOG_DIR/rl_grpo_${SCRIPT_BASENAME}_$TIMESTAMP.log"

# ================= 改动部分 2 =================

# 3. 将具体的 Python 命令定义为变量
# 注意：这里使用变量 $PYTHON_SCRIPT 替换了原来的死文件名
CMD="python -u $PYTHON_SCRIPT --config ./rl_configs/KuaiRec_big_qwen2_DIN_larger_lr_ttt.yaml"

# 4. 先将命令写入日志文件开头
echo "Execution Command: $CMD" > "$LOG_FILE"
echo "Start Time: $TIMESTAMP" >> "$LOG_FILE"
echo "------------------------------------------------------" >> "$LOG_FILE"

# 5. 执行命令，并将输出追加到日志
# 2>&1 将错误输出也合并，tee -a 追加到文件
echo "正在执行: $CMD"
$CMD 2>&1 | tee -a "$LOG_FILE"