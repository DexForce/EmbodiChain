#!/bin/bash
# 检查是否提供了数据目录参数
if [ $# -eq 0 ]; then
    echo "Usage: $0 <data_directory>"
    echo "Example: $0 /path/to/telecontrol_data"
    exit 1
fi
DATA_DIR="$1"
OUTPUT_DIR="$2"

# 遍历数据目录下的所有遥操数据文件夹
for tele_dir in "$DATA_DIR"/*/; do
    if [ -d "$tele_dir" ]; then
        # 从目录路径中提取时间戳作为文件名
        TIMESTAMP=$(basename "$tele_dir")
        OUTPUT_FILE="$OUTPUT_DIR/${TIMESTAMP}_compressed.hdf5"
        
        echo "Processing: $tele_dir"
        # 运行转换脚本
        python3 -m embodichain.utils.scripts.tele2hdf5.w1_telecontrol_to_hdf5 "$tele_dir" --output "$OUTPUT_FILE"
    fi
done

# 删除中间对齐JSON文件
ALIGNMENT_FILE="$OUTPUT_DIR/aligned_pose_record_*.json"
if ls $ALIGNMENT_FILE 1> /dev/null 2>&1; then
    rm -f $ALIGNMENT_FILE
    echo "🗑️  Deleted alignment files: $ALIGNMENT_FILE"
fi
